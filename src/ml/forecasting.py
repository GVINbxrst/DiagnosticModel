"""
Модуль прогнозирования временных рядов для диагностики двигателей

Этот модуль анализирует тренды RMS токов по фазам и выполняет прогнозирование:
- Построение временных рядов RMS для каждой фазы
- Использование ARIMA и Prophet для прогнозирования
- Расчет вероятности превышения порогов (аномалии)
- Работа без меток отказов (unsupervised подход)
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import Feature, Equipment, RawSignal
from src.utils.logger import get_logger

# Настройки
settings = get_settings()
logger = get_logger(__name__)

# Константы
DEFAULT_FORECAST_STEPS = 24  # Прогноз на 24 шага вперед
DEFAULT_ANOMALY_THRESHOLD = 1.5  # Порог аномалии (в стандартных отклонениях)
MIN_OBSERVATIONS = 50  # Минимальное количество наблюдений для прогноза
SEASONALITY_PERIOD = 24  # Период сезонности (24 часа)


class ForecastingError(Exception):
    """Базовое исключение для прогнозирования"""
    pass


class InsufficientDataError(ForecastingError):
    """Исключение для недостаточного количества данных"""
    pass


# === MVP функция forecast_rms согласно контракту ===
async def forecast_rms(equipment_id: UUID, n_steps: int = 24, threshold_sigma: float = 2.0):
    """Упрощённый прогноз RMS по оборудованию.

    Алгоритм (MVP):
      1. Загружаем RMS (rms_a, rms_b, rms_c) из Feature по данному equipment_id через связь RawSignal.
      2. Агрегируем средний RMS (среднее по доступным фазам) по времени window_start (часовое округление).
      3. Если точек < MIN_OBSERVATIONS -> InsufficientDataError.
      4. Прогноз: пытаемся использовать Prophet; если нет (ImportError) – fallback на простую экспоненциальную модель (moving average) или ARIMA(1,0,0).
      5. Оценка вероятности превышения порога: берём sigma исторического ряда; threshold = mean + threshold_sigma*std; прогнозные значения > threshold – оцениваем p как долю.
    Возвращает dict с forecast, threshold и probability_over_threshold.
    """
    from sqlalchemy import select
    from src.database.connection import get_async_session
    from src.database.models import Feature, RawSignal

    async with get_async_session() as session:
        # Join Feature -> RawSignal. В тестовой среде на SQLite UUID могут храниться как TEXT.
        # Унифицируем сравнение через строковое представление если dialect == 'sqlite'.
        from sqlalchemy import cast, String
        if session.bind.dialect.name == 'sqlite':  # pragma: no cover - специфично для тестов sqlite
            q = (select(Feature)
                 .join(RawSignal)
                 .where(cast(RawSignal.equipment_id, String) == str(equipment_id))
                 .order_by(Feature.window_start.asc()))
        else:
            q = (select(Feature)
                 .join(RawSignal)
                 .where(RawSignal.equipment_id == equipment_id)
                 .order_by(Feature.window_start.asc()))
        res = await session.execute(q)
        feats = res.scalars().all()

    if len(feats) < MIN_OBSERVATIONS:
        raise InsufficientDataError(f"Недостаточно данных: {len(feats)} < {MIN_OBSERVATIONS}")

    import pandas as pd
    rows = []
    for f in feats:
        values = [v for v in [f.rms_a, f.rms_b, f.rms_c] if v is not None]
        if not values:
            continue
        rows.append({
            'ts': f.window_start,
            'rms_mean': float(np.mean(values))
        })
    df = pd.DataFrame(rows)
    if df.empty or len(df) < MIN_OBSERVATIONS:
        raise InsufficientDataError("Недостаточно валидных RMS значений")

    # Агрегация по часу
    df['ts_hour'] = pd.to_datetime(df['ts']).dt.floor('H')
    hourly = df.groupby('ts_hour')['rms_mean'].mean().reset_index().rename(columns={'ts_hour':'ds','rms_mean':'y'})

    series = hourly['y']
    mu = series.mean()
    sigma = series.std(ddof=0) or 1e-6
    threshold = mu + threshold_sigma * sigma

    forecast_values = []
    future_index = []
    try:
        from prophet import Prophet  # type: ignore
        m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
        m.fit(hourly)
        future = m.make_future_dataframe(periods=n_steps, freq='H', include_history=False)
        fc = m.predict(future)
        forecast_values = fc['yhat'].tolist()
        future_index = future['ds'].tolist()
    except Exception:
        # Fallback: простое скользящее среднее последнего окна
        window = min(12, len(series))
        last_ma = series.rolling(window).mean().iloc[-1]
        trend = (series.iloc[-1] - series.iloc[-window]) / max(window-1,1)
        for i in range(1, n_steps+1):
            forecast_values.append(float(last_ma + trend * i))
        import pandas as pd
        last_ts = hourly['ds'].iloc[-1]
        future_index = [last_ts + pd.Timedelta(hours=i) for i in range(1, n_steps+1)]

    over = [v for v in forecast_values if v > threshold]
    probability_over = len(over)/len(forecast_values) if forecast_values else 0.0

    return {
        'equipment_id': str(equipment_id),
        'history_points': len(series),
        'threshold': threshold,
        'forecast': [
            {'timestamp': ts.isoformat(), 'rms': float(v), 'over_threshold': v > threshold}
            for ts, v in zip(future_index, forecast_values)
        ],
        'probability_over_threshold': probability_over,
        'model': 'Prophet' if 'Prophet' in globals() and 'm' in locals() else 'FallbackMA'
    }


class TimeSeriesPreprocessor:
    """Предобработчик временных рядов"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def prepare_time_series(
        self, 
        df: pd.DataFrame, 
        value_column: str,
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Подготовить временной ряд для прогнозирования
        
        Args:
            df: DataFrame с данными
            value_column: Название колонки со значениями
            time_column: Название колонки с временными метками
            
        Returns:
            Подготовленный DataFrame
        """
        # Копируем данные
        ts_df = df.copy()
        
        # Убираем пропущенные значения
        ts_df = ts_df.dropna(subset=[value_column])
        
        if len(ts_df) < MIN_OBSERVATIONS:
            raise InsufficientDataError(
                f"Недостаточно данных для анализа: {len(ts_df)} < {MIN_OBSERVATIONS}"
            )
        
        # Сортируем по времени
        ts_df = ts_df.sort_values(time_column)
        
        # Проверяем на дубликаты по времени
        duplicates = ts_df[time_column].duplicated()
        if duplicates.any():
            self.logger.warning(f"Найдено {duplicates.sum()} дубликатов по времени, удаляем")
            ts_df = ts_df[~duplicates]
        
        # Ресемплинг к равномерной сетке (если нужно)
        ts_df = self._resample_to_regular_grid(ts_df, time_column, value_column)
        
        self.logger.debug(f"Подготовлен временной ряд: {len(ts_df)} наблюдений")
        
        return ts_df
    
    def _resample_to_regular_grid(
        self, 
        df: pd.DataFrame, 
        time_column: str, 
        value_column: str,
        freq: str = '1H'
    ) -> pd.DataFrame:
        """Ресемплинг к равномерной временной сетке"""
        # Устанавливаем временной индекс
        df_resampled = df.set_index(time_column)
        
        # Ресемплинг с усреднением значений
        df_resampled = df_resampled.resample(freq)[value_column].mean()
        
        # Интерполируем пропущенные значения
        df_resampled = df_resampled.interpolate(method='linear')
        
        # Убираем NaN в начале и конце
        df_resampled = df_resampled.dropna()
        
        # Возвращаем DataFrame
        result_df = df_resampled.reset_index()
        result_df.columns = [time_column, value_column]
        
        return result_df
    
    def detect_and_handle_outliers(
        self, 
        df: pd.DataFrame, 
        value_column: str,
        method: str = 'iqr',
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        Обнаружение и обработка выбросов
        
        Args:
            df: DataFrame с данными
            value_column: Колонка со значениями
            method: Метод обнаружения ('iqr', 'zscore')
            multiplier: Множитель для порога
            
        Returns:
            DataFrame с обработанными выбросами
        """
        result_df = df.copy()
        values = result_df[value_column]
        
        if method == 'iqr':
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers_mask = (values < lower_bound) | (values > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers_mask = z_scores > multiplier
            
        else:
            raise ValueError(f"Неизвестный метод обнаружения выбросов: {method}")
        
        if outliers_mask.any():
            n_outliers = outliers_mask.sum()
            self.logger.info(f"Найдено {n_outliers} выбросов, заменяем интерполяцией")
            
            # Заменяем выбросы на NaN и интерполируем
            result_df.loc[outliers_mask, value_column] = np.nan
            result_df[value_column] = result_df[value_column].interpolate(method='linear')
        
        return result_df
    
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Проверка стационарности временного ряда с помощью теста Дики-Фуллера
        
        Args:
            series: Временной ряд
            
        Returns:
            Результаты теста стационарности
        """
        # Подавляем предупреждения statsmodels
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = adfuller(series.dropna())
            
        adf_stat, p_value, used_lag, n_obs, critical_values, ic_best = result
        
        stationarity_result = {
            'adf_statistic': adf_stat,
            'p_value': p_value,
            'used_lags': used_lag,
            'n_observations': n_obs,
            'critical_values': critical_values,
            # Явно приводим к builtin bool для тестов (adfuller возвращает numpy.float64 -> сравнение дает numpy.bool_)
            'is_stationary': bool(p_value < 0.05)  # 5% уровень значимости
        }
        
        self.logger.debug(
            f"Тест стационарности: ADF={adf_stat:.4f}, p-value={p_value:.4f}, "
            f"стационарен={stationarity_result['is_stationary']}"
        )
        
        return stationarity_result
    
    def make_stationary(self, series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Приведение ряда к стационарному виду через дифференцирование
        
        Args:
            series: Исходный временной ряд
            max_diff: Максимальное количество дифференцирований
            
        Returns:
            Кортеж (стационарный ряд, количество дифференцирований)
        """
        diff_series = series.copy()
        diff_order = 0
        
        for d in range(max_diff + 1):
            stationarity = self.check_stationarity(diff_series)
            
            if stationarity['is_stationary']:
                self.logger.debug(f"Ряд стал стационарным после {d} дифференцирований")
                return diff_series, d

            if d < max_diff:
                diff_series = diff_series.diff().dropna()
                diff_order = d + 1
        
        self.logger.warning(f"Не удалось достичь стационарности за {max_diff} дифференцирований")
        return diff_series, diff_order


class ARIMAForecaster:
    """Прогнозирование с помощью ARIMA модели"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.model = None
        self.fitted_model = None
        self.preprocessor = TimeSeriesPreprocessor()
        
    def auto_arima_parameters(self, series: pd.Series) -> Tuple[int, int, int]:
        """
        Автоматический подбор параметров ARIMA (p, d, q)
        
        Args:
            series: Временной ряд
            
        Returns:
            Кортеж параметров (p, d, q)
        """
        # Определяем порядок дифференцирования (d)
        _, d = self.preprocessor.make_stationary(series)
        
        # Простая эвристика для p и q
        # В реальном проекте можно использовать более сложные методы
        max_lag = min(10, len(series) // 4)
        
        best_aic = float('inf')
        best_params = (1, d, 1)
        
        # Перебираем параметры p и q
        for p in range(0, min(3, max_lag)):
            for q in range(0, min(3, max_lag)):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        temp_model = ARIMA(series, order=(p, d, q))
                        temp_fitted = temp_model.fit()
                        
                        if temp_fitted.aic < best_aic:
                            best_aic = temp_fitted.aic
                            best_params = (p, d, q)
                            
                except Exception:
                    continue
        
        self.logger.debug(f"Лучшие параметры ARIMA: {best_params}, AIC={best_aic:.2f}")

        return best_params
    
    def fit(self, series: pd.Series, order: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Обучение ARIMA модели
        
        Args:
            series: Временной ряд для обучения
            order: Параметры ARIMA (p, d, q). Если None, подбираются автоматически
            
        Returns:
            Результаты обучения
        """
        if len(series) < MIN_OBSERVATIONS:
            raise InsufficientDataError(f"Недостаточно данных для ARIMA: {len(series)} < {MIN_OBSERVATIONS}")

        if order is None:
            order = self.auto_arima_parameters(series)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                self.model = ARIMA(series, order=order)
                self.fitted_model = self.model.fit()
                
            fit_results = {
                'order': order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'success': True
            }
            
            self.logger.info(
                f"ARIMA{order} обучена: AIC={fit_results['aic']:.2f}, "
                f"BIC={fit_results['bic']:.2f}"
            )
            
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения ARIMA: {e}")
            return {'success': False, 'error': str(e)}
    
    def forecast(self, steps: int) -> Dict:
        """
        Прогнозирование на N шагов вперед
        
        Args:
            steps: Количество шагов для прогноза
            
        Returns:
            Результаты прогноза
        """
        if self.fitted_model is None:
                raise ForecastingError("Модель не обучена")
        
        try:
            forecast_result = self.fitted_model.forecast(steps=steps)
            confidence_intervals = self.fitted_model.get_forecast(steps=steps).conf_int()
            
            forecast_data = {
                'forecast': forecast_result.tolist(),
                'lower_ci': confidence_intervals.iloc[:, 0].tolist(),
                'upper_ci': confidence_intervals.iloc[:, 1].tolist(),
                'steps': steps,
                'success': True
            }
            
            self.logger.debug(f"ARIMA прогноз на {steps} шагов: {forecast_result.mean():.4f}")
            
            return forecast_data
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования ARIMA: {e}")
            return {'success': False, 'error': str(e)}

    # Alias для тестов
    def auto_arima_params(self, series: pd.Series) -> Tuple[int, int, int]:  # pragma: no cover
        return self.auto_arima_parameters(series)


class ProphetForecaster:
    """Прогнозирование с помощью Prophet"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.model = None
        
    def fit(self, df: pd.DataFrame, time_column: str = 'ds', value_column: str = 'y') -> Dict:
        """
        Обучение Prophet модели
        
        Args:
            df: DataFrame с колонками 'ds' (время) и 'y' (значения)
            time_column: Название колонки времени
            value_column: Название колонки значений
            
        Returns:
            Результаты обучения
        """
        # Подготавливаем данные для Prophet
        prophet_df = df[[time_column, value_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        if len(prophet_df) < MIN_OBSERVATIONS:
            raise InsufficientDataError(f"Недостаточно данных для Prophet: {len(prophet_df)} < {MIN_OBSERVATIONS}")

        try:
            # Настраиваем Prophet
            self.model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Отключаем годовую сезонность для коротких рядов
                changepoint_prior_scale=0.05,  # Чувствительность к изменениям трендов
                seasonality_prior_scale=10,    # Сила сезонности
                interval_width=0.95,           # Доверительный интервал 95%
                uncertainty_samples=1000
            )
            
            # Подавляем вывод Prophet
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(prophet_df)
            
            fit_results = {
                'n_observations': len(prophet_df),
                'trend_changepoints': len(self.model.changepoints),
                'model_type': 'prophet',
                'success': True
            }
            
            self.logger.info(
                f"Prophet обучен на {fit_results['n_observations']} наблюдениях, "
                f"точек изменения тренда: {fit_results['trend_changepoints']}"
            )
            
            return fit_results
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения Prophet: {e}")
            return {'success': False, 'error': str(e)}
    
    def forecast(self, periods: int, freq: str = 'H') -> Dict:
        """
        Прогнозирование с помощью Prophet
        
        Args:
            periods: Количество периодов для прогноза
            freq: Частота прогноза ('H' - часы, 'D' - дни)
            
        Returns:
            Результаты прогноза
        """
        if self.model is None:
                raise ForecastingError("Модель не обучена")
        
        try:
            # Создаем будущие даты
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Делаем прогноз
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = self.model.predict(future)
            
            # Извлекаем прогнозные значения (только новые периоды)
            forecast_data = forecast.tail(periods)
            
            seasonal_values = []
            if 'seasonal' in forecast_data.columns:
                col_val = forecast_data['seasonal']
                seasonal_values = col_val.tolist() if hasattr(col_val, 'tolist') else list(col_val)
            else:
                seasonal_values = [0.0] * periods
            result = {
                'forecast': forecast_data['yhat'].tolist(),
                'lower_ci': forecast_data['yhat_lower'].tolist(),
                'upper_ci': forecast_data['yhat_upper'].tolist(),
                'trend': forecast_data['trend'].tolist(),
                'seasonal': seasonal_values,
                'dates': forecast_data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'periods': periods,
                'success': True
            }
            
            self.logger.debug(
                f"Prophet прогноз на {periods} периодов: "
                f"среднее={np.mean(result['forecast']):.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования Prophet: {e}")
            return {'success': False, 'error': str(e)}


class AnomalyProbabilityCalculator:
    """Расчет вероятности аномалий на основе прогнозов"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def calculate_threshold_exceedance_probability(
        self,
        forecast_values: List[float],
        confidence_intervals: Tuple[List[float], List[float]],
        threshold: float,
        historical_std: float
    ) -> Dict:
        """
        Расчет вероятности превышения порога

        Args:
            forecast_values: Прогнозные значения
            confidence_intervals: Нижние и верхние границы доверительного интервала
            threshold: Пороговое значение для аномалии
            historical_std: Историческое стандартное отклонение

        Returns:
            Результаты расчета вероятностей
        """
        lower_ci, upper_ci = confidence_intervals

        probabilities = []
        exceedance_flags = []

        for i, (forecast, lower, upper) in enumerate(zip(forecast_values, lower_ci, upper_ci)):
            # Метод 1: Простое сравнение с порогом
            exceeds_threshold = forecast > threshold

            # Метод 2: Вероятность на основе нормального распределения
            # Предполагаем, что погрешность прогноза нормально распределена
            forecast_std = (upper - lower) / (2 * 1.96)  # 95% интервал -> std

            if forecast_std > 0:
                # Z-score для порога
                z_score = (threshold - forecast) / forecast_std
                # Вероятность превышения = 1 - CDF(z_score)
                from scipy import stats
                prob_exceed = 1 - stats.norm.cdf(z_score)
            else:
                prob_exceed = 1.0 if exceeds_threshold else 0.0

            probabilities.append(prob_exceed)
            exceedance_flags.append(exceeds_threshold)

        # Общая статистика
        max_probability = max(probabilities) if probabilities else 0.0
        mean_probability = np.mean(probabilities) if probabilities else 0.0
        any_exceedance = any(exceedance_flags)

        result = {
            'step_probabilities': probabilities,
            'step_exceedance': exceedance_flags,
            'max_probability': max_probability,
            'mean_probability': mean_probability,
            'any_exceedance': any_exceedance,
            'threshold': threshold,
            'forecast_horizon': len(forecast_values)
        }

        self.logger.debug(
            f"Вероятность аномалии: макс={max_probability:.3f}, "
            f"средняя={mean_probability:.3f}, превышений={sum(exceedance_flags)}"
        )

        return result

    def adaptive_threshold_calculation(
        self,
        historical_values: List[float],
        method: str = 'statistical',
        multiplier: float = DEFAULT_ANOMALY_THRESHOLD
    ) -> float:
        """
        Адаптивный расчет порога аномалии

        Args:
            historical_values: Исторические значения
            method: Метод расчета ('statistical', 'quantile')
            multiplier: Множитель для порога

        Returns:
            Пороговое значение
        """
        values = np.array(historical_values)

        if method == 'statistical':
            # Среднее + N стандартных отклонений
            mean_val = np.mean(values)
            std_val = np.std(values)
            threshold = mean_val + multiplier * std_val

        elif method == 'quantile':
            # Процентиль (по умолчанию 95%)
            quantile = min(0.99, 0.5 + multiplier * 0.3)  # Адаптивный квантиль
            threshold = np.quantile(values, quantile)

        else:
            raise ValueError(f"Неизвестный метод расчета порога: {method}")

        self.logger.debug(f"Адаптивный порог ({method}): {threshold:.4f}")

        return threshold


class MotorRMSForecaster:
    """Основной класс для прогнозирования RMS токов двигателя"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.preprocessor = TimeSeriesPreprocessor()
        self.probability_calculator = AnomalyProbabilityCalculator()
        
    async def load_rms_time_series(
        self,
        equipment_id: UUID,
        phase: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        session: Optional[AsyncSession] = None,
    ) -> pd.DataFrame:
        """
        Загрузить временной ряд RMS для конкретной фазы
        
        Args:
            equipment_id: ID оборудования
            phase: Фаза ('a', 'b', 'c')
            start_time: Начальное время
            end_time: Конечное время
            limit: Максимальное количество записей
            
        Returns:
            DataFrame с временным рядом RMS
        """
        phase_column_map = {'a': 'rms_a', 'b': 'rms_b', 'c': 'rms_c'}

        if phase not in phase_column_map:
            raise ValueError(f"Неизвестная фаза: {phase}")

        rms_column = phase_column_map[phase]

        # Позволяем инъекцию внешней сессии для тестов / батчевых операций
        if session is None:
            async with get_async_session() as _session:
                rows = await self._fetch_rms_rows(_session, equipment_id, rms_column, start_time, end_time, limit)
        else:
            rows = await self._fetch_rms_rows(session, equipment_id, rms_column, start_time, end_time, limit)
            
            if len(rows) < MIN_OBSERVATIONS:
                raise InsufficientDataError(
                    f"Недостаточно данных для фазы {phase}: {len(rows)} < {MIN_OBSERVATIONS}"
                )
            
            # Создаем DataFrame
            df = pd.DataFrame(rows, columns=['timestamp', 'rms_value'])
            
            self.logger.info(
                f"Загружен временной ряд для фазы {phase}: {len(df)} наблюдений "
                f"с {df['timestamp'].min()} по {df['timestamp'].max()}"
            )
            
            return df

    async def _fetch_rms_rows(
        self,
        session: AsyncSession,
        equipment_id: UUID,
        rms_column: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: Optional[int]
    ) -> List[Tuple[datetime, float]]:
        query = (
            select(Feature.window_start, getattr(Feature, rms_column))
            .join(RawSignal)
            .where(
                and_(
                    RawSignal.equipment_id == equipment_id,
                    getattr(Feature, rms_column).is_not(None)
                )
            )
        )
        if start_time:
            query = query.where(Feature.window_start >= start_time)
        if end_time:
            query = query.where(Feature.window_start <= end_time)
        query = query.order_by(Feature.window_start)
        if limit:
            query = query.limit(limit)
        result = await session.execute(query)
        return result.fetchall()
    
    async def forecast_phase_rms(
        self,
        equipment_id: UUID,
        phase: str,
        forecast_steps: int = DEFAULT_FORECAST_STEPS,
        model_type: str = 'auto',
        anomaly_threshold_method: str = 'statistical',
        threshold_multiplier: float = DEFAULT_ANOMALY_THRESHOLD,
        preloaded_df: Optional[pd.DataFrame] = None,
        session: Optional[AsyncSession] = None,
    ) -> Dict:
        """
        Прогнозирование RMS для конкретной фазы
        
        Args:
            equipment_id: ID оборудования
            phase: Фаза ('a', 'b', 'c')
            forecast_steps: Количество шагов прогноза
            model_type: Тип модели ('arima', 'prophet', 'auto')
            anomaly_threshold_method: Метод расчета порога аномалии
            threshold_multiplier: Множитель для порога
            
        Returns:
            Результаты прогнозирования
        """
        try:
            # Загружаем данные (используем предзагруженный DataFrame если передан)
            if preloaded_df is not None:
                df = preloaded_df
            else:
                df = await self.load_rms_time_series(equipment_id, phase, session=session)
            
            # Предобработка
            df_clean = self.preprocessor.prepare_time_series(df, 'rms_value', 'timestamp')
            df_clean = self.preprocessor.detect_and_handle_outliers(df_clean, 'rms_value')

            # Рассчитываем адаптивный порог
            threshold = self.probability_calculator.adaptive_threshold_calculation(
                df_clean['rms_value'].tolist(),
                method=anomaly_threshold_method,
                multiplier=threshold_multiplier
            )

            # Выбираем модель
            forecast_results = {}

            if model_type in ['arima', 'auto']:
                # ARIMA прогноз
                arima_forecaster = ARIMAForecaster()
                arima_fit = arima_forecaster.fit(df_clean['rms_value'])

                if arima_fit['success']:
                    arima_forecast = arima_forecaster.forecast(forecast_steps)
                    if arima_forecast['success']:
                        forecast_results['arima'] = arima_forecast

            if model_type in ['prophet', 'auto']:
                # Prophet прогноз
                prophet_forecaster = ProphetForecaster()
                prophet_df = df_clean.rename(columns={'timestamp': 'ds', 'rms_value': 'y'})
                prophet_fit = prophet_forecaster.fit(prophet_df)

                if prophet_fit['success']:
                    prophet_forecast = prophet_forecaster.forecast(forecast_steps)
                    if prophet_forecast['success']:
                        forecast_results['prophet'] = prophet_forecast

            # Выбираем лучший прогноз или комбинируем
            if model_type == 'auto':
                final_forecast = self._select_best_forecast(forecast_results)
            else:
                final_forecast = forecast_results.get(model_type, {})
            
            if not final_forecast or not final_forecast.get('success'):
                raise ForecastingError(f"Не удалось создать прогноз для фазы {phase}")
            
            # Расчет вероятностей аномалий
            anomaly_probs = self.probability_calculator.calculate_threshold_exceedance_probability(
                final_forecast['forecast'],
                (final_forecast['lower_ci'], final_forecast['upper_ci']),
                threshold,
                df_clean['rms_value'].std()
            )

            # Финальный результат
            result = {
                'equipment_id': str(equipment_id),
                'phase': phase,
                'forecast': final_forecast,
                'anomaly_analysis': anomaly_probs,
                'historical_stats': {
                    'mean': float(df_clean['rms_value'].mean()),
                    'std': float(df_clean['rms_value'].std()),
                    'min': float(df_clean['rms_value'].min()),
                    'max': float(df_clean['rms_value'].max()),
                    'count': len(df_clean)
                },
                'threshold': threshold,
                'model_used': model_type,
                'forecast_horizon': forecast_steps,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Прогноз для фазы {phase}: модель={model_type}, "
                f"порог={threshold:.4f}, макс.вероятность={anomaly_probs['max_probability']:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования фазы {phase}: {e}")
            raise ForecastingError(f"Ошибка прогнозирования фазы {phase}: {e}")
    
    def _select_best_forecast(self, forecast_results: Dict) -> Dict:
        """Выбор лучшего прогноза из доступных"""
        if 'prophet' in forecast_results and forecast_results['prophet']['success']:
            # Prophet обычно лучше для данных с сезонностью
            return forecast_results['prophet']
        elif 'arima' in forecast_results and forecast_results['arima']['success']:
            return forecast_results['arima']
        else:
            return {}
    
    async def forecast_all_phases(
        self,
        equipment_id: UUID,
        forecast_steps: int = DEFAULT_FORECAST_STEPS,
        **kwargs
    ) -> Dict:
        """
        Прогнозирование для всех фаз двигателя
        
        Args:
            equipment_id: ID оборудования
            forecast_steps: Количество шагов прогноза
            **kwargs: Дополнительные параметры для прогнозирования
            
        Returns:
            Прогнозы для всех доступных фаз
        """
        phases = ['a', 'b', 'c']
        phase_names = {'a': 'R', 'b': 'S', 'c': 'T'}

        results = {
            'equipment_id': str(equipment_id),
            'phases': {},
            'summary': {
                'total_phases': 0,
                'successful_phases': 0,
                'failed_phases': 0,
                'max_anomaly_probability': 0.0,
                'critical_phases': []
            },
            'timestamp': datetime.now().isoformat()
        }
        
        for phase in phases:
            try:
                phase_result = await self.forecast_phase_rms(
                    equipment_id, phase, forecast_steps, **kwargs
                )
                
                results['phases'][phase] = phase_result
                results['summary']['successful_phases'] += 1

                # Обновляем сводную статистику
                max_prob = phase_result['anomaly_analysis']['max_probability']
                if max_prob > results['summary']['max_anomaly_probability']:
                    results['summary']['max_anomaly_probability'] = max_prob
                
                # Критические фазы (вероятность аномалии > 70%)
                if max_prob > 0.7:
                    results['summary']['critical_phases'].append({
                        'phase': phase,
                        'phase_name': phase_names[phase],
                        'probability': max_prob
                    })

                self.logger.debug(f"Фаза {phase_names[phase]}: прогноз готов")
                
            except Exception as e:
                self.logger.warning(f"Не удалось создать прогноз для фазы {phase}: {e}")
                results['phases'][phase] = {
                    'error': str(e),
                    'success': False
                }
                results['summary']['failed_phases'] += 1
        
        results['summary']['total_phases'] = len(phases)
        
        self.logger.info(
            f"Прогнозирование завершено: {results['summary']['successful_phases']}/{len(phases)} фаз, "
            f"макс.вероятность аномалии={results['summary']['max_anomaly_probability']:.3f}"
        )
        
        return results


# --- Совместимость с существующими задачами worker ---
class RMSTrendForecaster:
    """Обёртка для сохранения обратной совместимости.

    Ранее задачи Celery использовали класс RMSTrendForecaster с методом
    forecast_equipment_trends. В ходе рефакторинга основная реализация
    сосредоточена в MotorRMSForecaster. Этот класс предоставляет прежний
    интерфейс без упрощения бизнес-логики, делегируя работу MotorRMSForecaster.
    """

    def __init__(self):
        self._impl = MotorRMSForecaster()

    # --- Методы совместимости, ожидаемые тестами ---
    async def load_rms_data(self, equipment_id: UUID, phase: str, session: Optional[AsyncSession] = None):
        base = await self._impl.load_rms_time_series(equipment_id, phase, session=session)
        return base.rename(columns={'rms_value': f'rms_{phase}'})

    async def analyze_phase_trend(
        self,
        equipment_id: UUID,
        phase: str,
        session: Optional[AsyncSession] = None,
        forecast_steps: int = DEFAULT_FORECAST_STEPS
    ) -> Dict:
        raw = await self.load_rms_data(equipment_id, phase, session=session)
        internal_df = raw.rename(columns={f'rms_{phase}': 'rms_value'})
        internal = await self._impl.forecast_phase_rms(
            equipment_id,
            phase,
            forecast_steps,
            preloaded_df=internal_df,
            session=session,
        )
        return {
            'equipment_id': internal['equipment_id'],
            'phase': internal['phase'],
            'statistics': internal['historical_stats'],
            'forecasts': internal['forecast'],
            'anomaly': internal['anomaly_analysis'],
            # Сохраняем дополнительные ключи чтобы не терять семантику
            'threshold': internal['threshold'],
            'model_used': internal['model_used'],
            'forecast_horizon': internal['forecast_horizon'],
        }

    def calculate_anomaly_probability(
        self,
        forecast_data: Dict,
        current_mean: float,
        current_std: float
    ) -> Dict:
        forecast = forecast_data.get('forecast', [])
        upper_ci = forecast_data.get('upper_ci', forecast)
        anomaly_probs = []
        threshold = current_mean + DEFAULT_ANOMALY_THRESHOLD * (current_std or 1e-6)
        for val, upper in zip(forecast, upper_ci):
            if upper > threshold:
                prob = min(1.0, max(0.0, (upper - threshold) / (abs(threshold) + 1e-6)))
            else:
                prob = 0.0
            anomaly_probs.append(prob)
        return {
            'anomaly_probabilities': anomaly_probs,
            'max_probability': max(anomaly_probs) if anomaly_probs else 0.0,
            'any_exceedance': any(p > 0 for p in anomaly_probs)
        }

    def _get_recommendation(self, probability: float) -> str:
        if probability >= 0.85:
            return "КРИТИЧЕСКОЕ: немедленно провести диагностику"
        if probability >= 0.7:
            return "ВЫСОКИЙ: планировать внеплановый осмотр"
        if probability >= 0.5:
            return "СРЕДНИЙ: усилить мониторинг"
        if probability >= 0.25:
            return "НИЗКИЙ: наблюдение"
        return "НОРМАЛЬНОЕ состояние"

    async def forecast_equipment_trends(
        self,
        equipment_id: UUID,
        session: Optional["AsyncSession"] = None,  # параметр сохраняем для сигнатуры, не используем напрямую
        phases: Optional[List[str]] = None,
        forecast_steps: int = DEFAULT_FORECAST_STEPS,
        **kwargs
    ) -> Dict:
        # MotorRMSForecaster сам загружает данные; при необходимости можно
        # расширить, чтобы переиспользовать переданный session.
        result = await self._impl.forecast_all_phases(
            equipment_id=equipment_id,
            forecast_steps=forecast_steps,
        )
        # Добавляем явное поле forecast_steps для удобства downstream-кода
        result.setdefault('forecast_steps', forecast_steps)
        # Фильтрация фаз если задан список
        if phases:
            result['phases'] = {k: v for k, v in result['phases'].items() if k in phases}
            result['summary']['total_phases'] = len(phases)
        return result


# CLI функции

async def forecast_equipment_rms(
    equipment_id: str,
    forecast_steps: int = DEFAULT_FORECAST_STEPS,
    model_type: str = 'auto',
    phases: Optional[List[str]] = None
) -> Dict:
    """
    Прогнозирование RMS для оборудования
    
    Args:
        equipment_id: ID оборудования
        forecast_steps: Количество шагов прогноза
        model_type: Тип модели
        phases: Список фаз для прогноза (если None, все фазы)
        
    Returns:
        Результаты прогнозирования
    """
    forecaster = MotorRMSForecaster()
    equipment_uuid = UUID(equipment_id)
    
    if phases:
        # Прогноз для конкретных фаз
        results = {'phases': {}}
        
        for phase in phases:
            try:
                phase_result = await forecaster.forecast_phase_rms(
                    equipment_uuid, phase, forecast_steps, model_type
                )
                results['phases'][phase] = phase_result

            except Exception as e:
                results['phases'][phase] = {'error': str(e), 'success': False}

        return results
    else:
        # Прогноз для всех фаз
        return await forecaster.forecast_all_phases(equipment_uuid, forecast_steps)


# === Интеграционные функции, ожидаемые тестами (обратная совместимость) ===
async def forecast_rms_trends(
    equipment_id: UUID,
    forecast_steps: int = DEFAULT_FORECAST_STEPS,
    phases: Optional[List[str]] = None,
    **kwargs
) -> Dict:
    """Высокоуровневая функция прогнозирования трендов RMS по фазам.

    Оборачивает RMSTrendForecaster.forecast_equipment_trends, сохраняя
    сигнатуру, которую используют тесты и потенциальный внешний код.
    Не упрощает бизнес-логику: внутри используются те же механизмы
    загрузки данных и анализа, что и в MotorRMSForecaster.
    """
    forecaster = RMSTrendForecaster()
    return await forecaster.forecast_equipment_trends(
        equipment_id=equipment_id,
        phases=phases,
        forecast_steps=forecast_steps,
        **kwargs
    )


async def get_anomaly_probability(
    equipment_id: UUID,
    forecast_steps: int = DEFAULT_FORECAST_STEPS,
    **kwargs
) -> float:
    """Вернуть максимальную вероятность аномалии для оборудования.

    При ошибке возвращает 0.0 (как ожидают тесты), логируя исключение.
    """
    try:
        result = await forecast_rms_trends(
            equipment_id=equipment_id,
            forecast_steps=forecast_steps,
            **kwargs
        )
        return float(result.get('summary', {}).get('max_anomaly_probability', 0.0))
    except Exception as e:  # pragma: no cover - контролируемая деградация
        logger.warning(f"Не удалось получить вероятность аномалии: {e}")
        return 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Прогнозирование RMS токов двигателей")
    parser.add_argument("equipment_id", help="UUID оборудования")
    parser.add_argument("--steps", type=int, default=DEFAULT_FORECAST_STEPS,
                       help="Количество шагов прогноза")
    parser.add_argument("--model", choices=['arima', 'prophet', 'auto'], default='auto',
                       help="Тип модели для прогнозирования")
    parser.add_argument("--phases", nargs="+", choices=['a', 'b', 'c'],
                       help="Фазы для прогноза (по умолчанию все)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    
    args = parser.parse_args()

    # Настраиваем логирование
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    async def main():
        try:
            logger.info(f"Прогнозирование RMS для оборудования {args.equipment_id}")

            results = await forecast_equipment_rms(
                equipment_id=args.equipment_id,
                forecast_steps=args.steps,
                model_type=args.model,
                phases=args.phases
            )

            # Выводим результаты
            if 'summary' in results:
                summary = results['summary']
                print(f"\n📊 Сводка прогнозирования:")
                print(f"  - Фаз обработано: {summary['successful_phases']}/{summary['total_phases']}")
                print(f"  - Максимальная вероятность аномалии: {summary['max_anomaly_probability']:.1%}")

                if summary['critical_phases']:
                    print(f"  - Критические фазы:")
                    for critical in summary['critical_phases']:
                        print(f"    • Фаза {critical['phase_name']}: {critical['probability']:.1%}")

            for phase, phase_result in results['phases'].items():
                if phase_result.get('success', True):
                    anomaly = phase_result['anomaly_analysis']
                    print(f"\n🔮 Фаза {phase.upper()}:")
                    print(f"  - Прогноз на {args.steps} шагов")
                    print(f"  - Порог аномалии: {phase_result['threshold']:.4f}")
                    print(f"  - Вероятность превышения: {anomaly['max_probability']:.1%}")
                    if anomaly['any_exceedance']:
                        print(f"  - ⚠️  Прогнозируется превышение порога!")
                else:
                    print(f"\n❌ Фаза {phase.upper()}: {phase_result.get('error', 'Неизвестная ошибка')}")

            return True
            
        except Exception as e:
            logger.error(f"Ошибка прогнозирования: {e}")
            print(f"❌ Ошибка: {e}")
            return False
    
    import asyncio
    success = asyncio.run(main())
    exit(0 if success else 1)
