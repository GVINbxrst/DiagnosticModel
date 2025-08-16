"""Тесты для модуля прогнозирования временных рядов"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.ml.forecasting import (
    TimeSeriesPreprocessor,
    ARIMAForecaster,
    ProphetForecaster,
    RMSTrendForecaster,
    ForecastingError,
    InsufficientDataError,
    forecast_rms_trends,
    get_anomaly_probability,
    DEFAULT_FORECAST_STEPS
)


@pytest.fixture
def sample_time_series():
    """Создает образец временного ряда для тестирования"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    values = np.random.normal(10, 2, 100) + np.sin(np.arange(100) * 0.1) * 3
    return pd.DataFrame({
        'timestamp': dates,
        'value': values
    })


@pytest.fixture
def rms_data():
    """Создает образец данных RMS"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='h')
    rms_a = np.random.normal(5.0, 0.5, 200) + np.sin(np.arange(200) * 0.05) * 0.3
    return pd.DataFrame({
        'timestamp': dates,
        'rms_a': rms_a
    })


class TestTimeSeriesPreprocessor:
    """Тесты для предобработчика временных рядов"""

    def test_prepare_time_series_success(self, sample_time_series):
        """Тест успешной подготовки временного ряда"""
        preprocessor = TimeSeriesPreprocessor()

        result = preprocessor.prepare_time_series(sample_time_series, 'value')

        assert len(result) > 0
        assert 'timestamp' in result.columns
        assert 'value' in result.columns
        assert result['value'].notna().all()

    def test_prepare_time_series_insufficient_data(self):
        """Тест с недостаточным количеством данных"""
        preprocessor = TimeSeriesPreprocessor()

        # Создаем слишком малый датасет
        small_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='h'),
            'value': np.random.normal(10, 2, 10)
        })

        with pytest.raises(InsufficientDataError):
            preprocessor.prepare_time_series(small_df, 'value')

    def test_detect_outliers_iqr(self, sample_time_series):
        """Тест обнаружения выбросов методом IQR"""
        preprocessor = TimeSeriesPreprocessor()

        # Добавляем выброс
        outlier_data = sample_time_series.copy()
        outlier_data.loc[50, 'value'] = 1000  # Очевидный выброс

        result = preprocessor.detect_and_handle_outliers(outlier_data, 'value', method='iqr')

        # Выброс должен быть исправлен
        assert result.loc[50, 'value'] != 1000
        assert not np.isnan(result.loc[50, 'value'])

    def test_check_stationarity(self, sample_time_series):
        """Тест проверки стационарности"""
        preprocessor = TimeSeriesPreprocessor()

        result = preprocessor.check_stationarity(sample_time_series['value'])

        assert 'adf_statistic' in result
        assert 'p_value' in result
        assert 'is_stationary' in result
        assert isinstance(result['is_stationary'], bool)

    def test_make_stationary(self, sample_time_series):
        """Тест приведения к стационарности"""
        preprocessor = TimeSeriesPreprocessor()

        # Создаем нестационарный ряд (с трендом)
        trend_series = sample_time_series['value'] + np.arange(len(sample_time_series)) * 0.1

        stationary_series, diff_order = preprocessor.make_stationary(trend_series)

        assert len(stationary_series) > 0
        assert diff_order >= 0
        assert diff_order <= 2


class TestARIMAForecaster:
    """Тесты для ARIMA прогнозирования"""

    def test_auto_arima_params(self, sample_time_series):
        """Тест автоподбора параметров ARIMA"""
        forecaster = ARIMAForecaster()

        p, d, q = forecaster.auto_arima_params(sample_time_series['value'])

        assert isinstance(p, int) and p >= 0
        assert isinstance(d, int) and d >= 0
        assert isinstance(q, int) and q >= 0

    def test_fit_and_forecast(self, sample_time_series):
        """Тест обучения и прогнозирования ARIMA"""
        forecaster = ARIMAForecaster()

        # Обучение
        metrics = forecaster.fit(sample_time_series['value'])

        assert metrics['success'] is True
        assert 'aic' in metrics
        assert 'order' in metrics

        # Прогнозирование
        forecast_result = forecaster.forecast(steps=10)

        assert 'forecast' in forecast_result
        assert 'lower_ci' in forecast_result
        assert 'upper_ci' in forecast_result
        assert len(forecast_result['forecast']) == 10

    def test_fit_insufficient_data(self):
        """Тест с недостаточными данными для ARIMA"""
        forecaster = ARIMAForecaster()

        small_series = pd.Series(np.random.normal(0, 1, 10))

        with pytest.raises(InsufficientDataError):
            forecaster.fit(small_series)


class TestProphetForecaster:
    """Тесты для Prophet прогнозирования"""

    def test_fit_and_forecast(self, sample_time_series):
        """Тест обучения и прогнозирования Prophet"""
        forecaster = ProphetForecaster()

        # Подготавливаем данные для Prophet
        prophet_df = sample_time_series.copy()
        prophet_df.columns = ['ds', 'y']

        # Обучение
        metrics = forecaster.fit(prophet_df)

        assert metrics['success'] is True
        assert 'n_observations' in metrics
        assert 'model_type' in metrics

        # Прогнозирование
        forecast_result = forecaster.forecast(periods=10)

        assert 'forecast' in forecast_result
        assert 'lower_ci' in forecast_result
        assert 'upper_ci' in forecast_result
        assert 'dates' in forecast_result
        assert len(forecast_result['forecast']) == 10

    def test_fit_insufficient_data(self):
        """Тест с недостаточными данными для Prophet"""
        forecaster = ProphetForecaster()

        small_df = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=10, freq='h'),
            'y': np.random.normal(0, 1, 10)
        })

        with pytest.raises(InsufficientDataError):
            forecaster.fit(small_df)


class TestRMSTrendForecaster:
    """Тесты для основного класса прогнозирования RMS"""

    @pytest.fixture
    def mock_session(self):
        """Мок сессии БД"""
        session = AsyncMock()

        # Мокаем результат запроса к БД
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            (datetime.now() - timedelta(hours=i), np.random.normal(5.0, 0.5))
            for i in range(100, 0, -1)
        ]
        session.execute.return_value = mock_result

        return session

    @pytest.mark.asyncio
    async def test_load_rms_data(self, mock_session):
        """Тест загрузки данных RMS"""
        forecaster = RMSTrendForecaster()
        equipment_id = uuid4()

        df = await forecaster.load_rms_data(equipment_id, 'a', mock_session)

        assert len(df) > 0
        assert 'timestamp' in df.columns
        assert 'rms_a' in df.columns

    @pytest.mark.asyncio
    async def test_analyze_phase_trend(self, mock_session):
        """Тест анализа тренда для одной фазы"""
        forecaster = RMSTrendForecaster()
        equipment_id = uuid4()

        with patch.object(forecaster, 'load_rms_data') as mock_load:
            # Мокаем загрузку данных
            mock_load.return_value = pd.DataFrame({
                'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),
                'rms_a': np.random.normal(5.0, 0.5, 100)
            })

            result = await forecaster.analyze_phase_trend(
                equipment_id, 'a', mock_session, forecast_steps=12
            )

            assert 'phase' in result
            assert 'equipment_id' in result
            assert 'statistics' in result
            assert 'forecasts' in result
            assert result['phase'] == 'a'

    def test_calculate_anomaly_probability(self):
        """Тест расчета вероятности аномалии"""
        forecaster = RMSTrendForecaster()

        # Тестовые данные прогноза
        forecast_data = {
            'forecast': [5.0, 6.0, 8.0, 12.0, 15.0],  # Возрастающий тренд
            'upper_ci': [6.0, 7.0, 9.0, 13.0, 16.0]
        }

        result = forecaster.calculate_anomaly_probability(
            forecast_data,
            current_mean=5.0,
            current_std=1.0
        )

        assert 'anomaly_probabilities' in result
        assert 'max_probability' in result
        assert len(result['anomaly_probabilities']) == 5
        assert 0 <= result['max_probability'] <= 1

    def test_get_recommendation(self):
        """Тест получения рекомендаций"""
        forecaster = RMSTrendForecaster()

        # Тестируем разные уровни риска
        assert "КРИТИЧЕСКОЕ" in forecaster._get_recommendation(0.9)
        assert "ВЫСОКИЙ" in forecaster._get_recommendation(0.7)
        assert "СРЕДНИЙ" in forecaster._get_recommendation(0.5)
        assert "НИЗКИЙ" in forecaster._get_recommendation(0.3)
        assert "НОРМАЛЬНОЕ" in forecaster._get_recommendation(0.1)


class TestIntegrationFunctions:
    """Тесты интеграционных функций"""

    @pytest.mark.asyncio
    async def test_forecast_rms_trends(self):
        """Тест главной функции прогнозирования"""
        equipment_id = uuid4()

        with patch('src.ml.forecasting.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Мокаем результат запроса
            mock_result = Mock()
            mock_result.fetchall.return_value = [
                (datetime.now() - timedelta(hours=i), np.random.normal(5.0, 0.5))
                for i in range(100, 0, -1)
            ]
            mock_session.execute.return_value = mock_result

            # Тестируем с пропуском реального выполнения
            with patch.object(RMSTrendForecaster, 'forecast_equipment_trends') as mock_forecast:
                mock_forecast.return_value = {
                    'equipment_id': str(equipment_id),
                    'summary': {'max_anomaly_probability': 0.3}
                }

                result = await forecast_rms_trends(equipment_id, forecast_steps=12)

                assert 'equipment_id' in result
                assert 'summary' in result

    @pytest.mark.asyncio
    async def test_get_anomaly_probability(self):
        """Тест упрощенной функции получения вероятности аномалии"""
        equipment_id = uuid4()

        with patch('src.ml.forecasting.forecast_rms_trends') as mock_forecast:
            mock_forecast.return_value = {
                'summary': {'max_anomaly_probability': 0.75}
            }

            probability = await get_anomaly_probability(equipment_id, 24)

            assert probability == 0.75

    @pytest.mark.asyncio
    async def test_get_anomaly_probability_error_handling(self):
        """Тест обработки ошибок в функции получения вероятности аномалии"""
        equipment_id = uuid4()

        with patch('src.ml.forecasting.forecast_rms_trends') as mock_forecast:
            mock_forecast.side_effect = Exception("Тестовая ошибка")

            probability = await get_anomaly_probability(equipment_id, 24)

            assert probability == 0.0


class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_forecasting_error(self):
        """Тест базового исключения прогнозирования"""
        with pytest.raises(ForecastingError):
            raise ForecastingError("Тестовая ошибка")

    def test_insufficient_data_error(self):
        """Тест исключения недостаточных данных"""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("Недостаточно данных")

    def test_arima_forecast_without_fit(self):
        """Тест прогнозирования без обучения модели"""
        forecaster = ARIMAForecaster()

        with pytest.raises(ForecastingError):
            forecaster.forecast(10)

    def test_prophet_forecast_without_fit(self):
        """Тест прогнозирования Prophet без обучения"""
        forecaster = ProphetForecaster()

        with pytest.raises(ForecastingError):
            forecaster.forecast(10)


# Параметризованные тесты
@pytest.mark.parametrize("method,multiplier", [
    ('iqr', 1.5),
    ('iqr', 2.0),
    ('zscore', 2.0),
    ('zscore', 3.0),
])
def test_outlier_detection_methods(sample_time_series, method, multiplier):
    """Параметризованный тест методов обнаружения выбросов"""
    preprocessor = TimeSeriesPreprocessor()

    # Добавляем выброс
    outlier_data = sample_time_series.copy()
    outlier_data.loc[50, 'value'] = 1000

    result = preprocessor.detect_and_handle_outliers(
        outlier_data, 'value', method=method, multiplier=multiplier
    )

    assert len(result) == len(outlier_data)
    assert result['value'].notna().all()


@pytest.mark.parametrize("forecast_steps", [1, 12, 24, 48])
def test_forecast_steps_variations(sample_time_series, forecast_steps):
    """Параметризованный тест различных горизонтов прогнозирования"""
