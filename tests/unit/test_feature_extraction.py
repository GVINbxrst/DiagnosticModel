"""
Unit тесты для модуля извлечения признаков
Тестирование обработки сигналов и извлечения статистических/частотных характеристик
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.data_processing.feature_extraction import (
    SignalPreprocessor,
    StatisticalFeatureExtractor,
    FrequencyFeatureExtractor,
    FeatureExtractor,
    InsufficientDataError,
    SignalProcessingError,
    DEFAULT_SAMPLE_RATE
)


class TestSignalPreprocessor:
    """Тесты предобработчика сигналов"""

    @pytest.fixture
    def preprocessor(self):
        return SignalPreprocessor()

    def test_clean_signal_without_nan(self, preprocessor):
        """Тест обработки сигнала без NaN"""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = preprocessor.clean_and_interpolate_signal(signal)

        np.testing.assert_array_equal(result, signal)

    def test_interpolate_internal_nan(self, preprocessor):
        """Тест интерполяции внутренних NaN"""
        signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        result = preprocessor.clean_and_interpolate_signal(signal)

        # Проверяем, что NaN заменился интерполированным значением
        assert not np.any(np.isnan(result))
        assert len(result) == len(signal)
        assert result[2] == 3.0  # Линейная интерполяция между 2 и 4

    def test_trim_leading_trailing_nan(self, preprocessor):
        """Тест обрезки NaN в начале и конце"""
        signal = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, np.nan, np.nan])

        result = preprocessor.clean_and_interpolate_signal(signal)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_complex_nan_pattern(self, preprocessor):
        """Тест сложного паттерна с NaN"""
        signal = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, 5.0, np.nan])

        result = preprocessor.clean_and_interpolate_signal(signal)

        # Ожидаем: [1.0, 2.0, 3.0, 4.0, 5.0] после обрезки и интерполяции
        assert not np.any(np.isnan(result))
        assert len(result) == 5
        assert result[0] == 1.0
        assert result[-1] == 5.0

    def test_too_many_nan_values(self, preprocessor):
        """Тест сигнала со слишком большим количеством NaN"""
        signal = np.array([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.0])

        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(signal, max_nan_ratio=0.5)

    def test_empty_signal(self, preprocessor):
        """Тест пустого сигнала"""
        signal = np.array([])

        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(signal)

    def test_all_nan_signal(self, preprocessor):
        """Тест сигнала из одних NaN"""
        signal = np.array([np.nan, np.nan, np.nan])

        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(signal)


class TestStatisticalFeatureExtractor:
    """Тесты извлечения статистических признаков"""

    @pytest.fixture
    def extractor(self):
        return StatisticalFeatureExtractor()

    def test_extract_basic_features(self, extractor):
        """Тест извлечения базовых статистических признаков"""
        # Создаем тестовый сигнал с известными свойствами
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        features = extractor.extract_statistical_features(signal)

        # Проверяем основные статистические характеристики
        assert features['mean'] == 3.0
        assert features['min'] == 1.0
        assert features['max'] == 5.0
        assert abs(features['std'] - np.std(signal)) < 1e-6

        # RMS должен быть больше среднего для положительных значений
        expected_rms = np.sqrt(np.mean(signal**2))
        assert abs(features['rms'] - expected_rms) < 1e-6

        # Crest factor = max / rms
        expected_crest = 5.0 / expected_rms
        assert abs(features['crest_factor'] - expected_crest) < 1e-6

    def test_symmetric_signal_skewness(self, extractor):
        """Тест асимметрии для симметричного сигнала"""
        # Симметричный сигнал должен иметь skewness близкий к 0
        signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        features = extractor.extract_statistical_features(signal)

        assert abs(features['skewness']) < 1e-10

    def test_normal_distribution_kurtosis(self, extractor):
        """Тест эксцесса для нормального распределения"""
        # Генерируем большой сигнал из нормального распределения
        np.random.seed(42)
        signal = np.random.normal(0, 1, 10000)

        features = extractor.extract_statistical_features(signal)

        # Kurtosis нормального распределения должен быть близок к 0
        assert abs(features['kurtosis']) < 0.1

    def test_zero_rms_signal(self, extractor):
        """Тест сигнала с нулевым RMS"""
        signal = np.array([0.0, 0.0, 0.0, 0.0])

        features = extractor.extract_statistical_features(signal)

        assert features['rms'] == 0.0
        assert features['crest_factor'] == 0.0  # Должен обрабатывать деление на ноль

    def test_single_value_signal(self, extractor):
        """Тест сигнала из одного значения"""
        signal = np.array([5.0])

        features = extractor.extract_statistical_features(signal)

        assert features['mean'] == 5.0
        assert features['min'] == 5.0
        assert features['max'] == 5.0
        assert features['std'] == 0.0


class TestFrequencyFeatureExtractor:
    """Тесты извлечения частотных признаков"""

    @pytest.fixture
    def extractor(self):
        return FrequencyFeatureExtractor(sample_rate=1000)  # Простая частота для тестов

    def test_extract_fft_from_sine_wave(self, extractor):
        """Тест FFT анализа синусоидального сигнала"""
        # Создаем синусоиду с частотой 50 Гц
        sample_rate = 1000
        duration = 1.0  # 1 секунда
        frequency = 50.0  # 50 Гц

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)

        features = extractor.extract_fft_features(signal, window_size=512)

        # Проверяем, что доминантная частота близка к 50 Гц
        assert abs(features['dominant_frequency'] - frequency) < 5.0  # Погрешность 5 Гц

        # Проверяем наличие пиков
        assert len(features['peaks']) > 0
        assert features['peaks'][0]['frequency'] > 40  # Первый пик должен быть около 50 Гц
        assert features['peaks'][0]['frequency'] < 60

    def test_fft_multiple_frequencies(self, extractor):
        """Тест FFT с несколькими частотными компонентами"""
        sample_rate = 1000
        duration = 1.0

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Смесь 50 Гц и 100 Гц
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)

    features = extractor.extract_fft_features(signal, window_size=512, top_peaks=5)

        # Должно быть найдено несколько пиков
        assert len(features['peaks']) >= 2

    def test_fft_top_peaks_limit(self, extractor):
        """Убеждаемся что возвращается не больше top_peaks пиков"""
        sample_rate = 1000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Микс многих частот
        sig = sum(np.sin(2*np.pi*f*t) for f in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        features = extractor.extract_fft_features(sig, window_size=512, top_peaks=10)
        assert len(features['peaks']) <= 10

    def test_fft_window_auto_reduce(self, extractor):
        """Если сигнал короче окна 4096, окно должно уменьшаться без ошибки"""
        sample_rate = 1000
        t = np.linspace(0, 0.2, int(sample_rate * 0.2), endpoint=False)
        sig = np.sin(2*np.pi*60*t)
        features = extractor.extract_fft_features(sig, window_size=4096)
        assert 'peaks' in features
        assert len(features['peaks']) > 0


class TestNanPolicy:
    def test_nan_ratio_exceeds_threshold(self):
        from src.data_processing.feature_extraction import SignalPreprocessor, InsufficientDataError, MAX_NAN_RATIO
        prep = SignalPreprocessor()
        # 30% NaN (>20%)
        data = np.array([1.0, np.nan, 2.0, np.nan, 3.0, np.nan, 4.0], dtype=float)
        with pytest.raises(InsufficientDataError):
            prep.clean_and_interpolate_signal(data, max_nan_ratio=MAX_NAN_RATIO)

    def test_nan_ratio_under_threshold_interpolated(self):
        from src.data_processing.feature_extraction import SignalPreprocessor, MAX_NAN_RATIO
        prep = SignalPreprocessor()
        # ~16.6% NaN (<20%)
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], dtype=float)
        cleaned = prep.clean_and_interpolate_signal(data, max_nan_ratio=MAX_NAN_RATIO)
        assert not np.isnan(cleaned).any()
        assert len(cleaned) == 6 or len(cleaned) == 5

    def test_short_signal_fft(self, extractor):
        """Тест FFT для короткого сигнала"""
        # Сигнал короче стандартного окна FFT
        signal = np.sin(2 * np.pi * 0.1 * np.arange(100))

        features = extractor.extract_fft_features(signal, window_size=512)

        # Должен успешно обработаться с предупреждением
        assert 'frequencies' in features
        assert 'magnitude_spectrum' in features
        assert features['window_size'] == 100  # Размер окна адаптировался

    def test_spectral_features(self, extractor):
        """Тест вычисления спектральных характеристик"""
        # Белый шум для проверки спектральных характеристик
        np.random.seed(42)
        signal = np.random.normal(0, 1, 1024)

        features = extractor.extract_fft_features(signal)

        # Проверяем наличие спектральных характеристик
        assert 'spectral_centroid' in features
        assert 'spectral_bandwidth' in features
        assert 'spectral_rolloff' in features
        assert 'spectral_energy' in features

        # Все значения должны быть положительными
        assert features['spectral_centroid'] > 0
        assert features['spectral_bandwidth'] > 0
        assert features['spectral_rolloff'] > 0
        assert features['spectral_energy'] > 0


class TestFeatureExtractor:
    """Тесты основного класса извлечения признаков"""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(sample_rate=1000)

    def test_extract_features_single_phase(self, extractor):
        """Тест извлечения признаков из одной фазы"""
        # Создаем тестовый сигнал
        t = np.linspace(0, 1, 1000)
        phase_a = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.normal(0, 1, 1000)

        window_start = datetime.now()
        window_end = window_start + timedelta(seconds=1)

        features = extractor.extract_features_from_phases(
            phase_a=phase_a,
            window_start=window_start,
            window_end=window_end
        )

        # Проверяем структуру результата
        assert 'phases' in features
        assert 'a' in features['phases']
        assert features['phases']['a'] is not None
        assert features['phases']['b'] is None
        assert features['phases']['c'] is None

        # Проверяем наличие статистических признаков
        phase_features = features['phases']['a']
        assert 'statistical' in phase_features
        assert 'frequency' in phase_features

        stat_features = phase_features['statistical']
        assert 'rms' in stat_features
        assert 'crest_factor' in stat_features
        assert 'skewness' in stat_features
        assert 'kurtosis' in stat_features

        # Проверяем частотные признаки
        freq_features = phase_features['frequency']
        assert 'peaks' in freq_features
        assert 'dominant_frequency' in freq_features
        assert 'spectral_centroid' in freq_features

    def test_extract_features_three_phases(self, extractor):
        """Тест извлечения признаков из трех фаз"""
        # Создаем сигналы для трех фаз с разными характеристиками
        t = np.linspace(0, 1, 1000)
        phase_a = np.sin(2 * np.pi * 50 * t)
        phase_b = np.sin(2 * np.pi * 50 * t + 2*np.pi/3)  # Сдвиг на 120°
        phase_c = np.sin(2 * np.pi * 50 * t + 4*np.pi/3)  # Сдвиг на 240°

        features = extractor.extract_features_from_phases(
            phase_a=phase_a,
            phase_b=phase_b,
            phase_c=phase_c
        )

        # Все три фазы должны быть обработаны
        assert features['phases']['a'] is not None
        assert features['phases']['b'] is not None
        assert features['phases']['c'] is not None

        # Проверяем сводную информацию
        assert features['summary']['processed_phases'] == 3
        assert features['summary']['total_phases'] == 3

    def test_extract_features_with_nan(self, extractor):
        """Тест извлечения признаков с NaN значениями"""
        # Создаем сигнал с пропусками
        t = np.linspace(0, 1, 1000)
        phase_a = np.sin(2 * np.pi * 50 * t)

        # Добавляем NaN в середину
        phase_a[400:450] = np.nan

        features = extractor.extract_features_from_phases(phase_a=phase_a)

        # Должен успешно обработаться
        assert features['phases']['a'] is not None

        # Проверяем информацию о качестве данных
        data_quality = features['phases']['a']['data_quality']
        assert data_quality['original_length'] == 1000
        assert data_quality['processed_length'] < 1000  # После обрезки NaN
        assert data_quality['nan_ratio'] > 0

    def test_insufficient_data_error(self, extractor):
        """Тест обработки недостаточного количества данных"""
        # Очень короткий сигнал
        phase_a = np.array([1.0, 2.0])

        with pytest.raises(InsufficientDataError):
            extractor.extract_features_from_phases(phase_a=phase_a)

    def test_all_phases_none(self, extractor):
        """Тест случая, когда все фазы пустые"""
        with pytest.raises(InsufficientDataError):
            extractor.extract_features_from_phases(
                phase_a=None,
                phase_b=None,
                phase_c=None
            )

    @pytest.mark.asyncio
    async def test_save_features_to_db(self, extractor):
        """Тест сохранения признаков в базу данных"""
        # Мокаем сессию базы данных
        mock_session = AsyncMock()

        # Создаем тестовые признаки
        features = {
            'phases': {
                'a': {
                    'statistical': {
                        'rms': 1.5, 'crest_factor': 1.2, 'kurtosis': 0.1, 'skewness': 0.05,
                        'mean': 1.0, 'std': 0.5, 'min': 0.0, 'max': 2.0
                    },
                    'frequency': {
                        'peaks': [{'frequency': 50.0, 'amplitude': 1.0}],
                        'spectral_centroid': 75.0
                    }
                },
                'b': None,
                'c': None
            }
        }

        raw_signal_id = uuid4()
        window_start = datetime.now()
        window_end = window_start + timedelta(seconds=1)

        # Вызываем метод сохранения
        feature_id = await extractor._save_features_to_db(
            mock_session, raw_signal_id, features, window_start, window_end
        )

        # Проверяем, что объект был добавлен в сессию
        assert mock_session.add.called
        assert mock_session.flush.called

        # Получаем добавленный объект
        added_feature = mock_session.add.call_args[0][0]

        assert added_feature.raw_id == raw_signal_id
        assert added_feature.window_start == window_start
        assert added_feature.window_end == window_end
        assert added_feature.rms_a == 1.5
        assert added_feature.crest_a == 1.2
        assert added_feature.rms_b is None  # Фаза B не обработана


@pytest.mark.integration
class TestFeatureExtractionIntegration:
    """Интеграционные тесты извлечения признаков"""

    @pytest.fixture
    def sample_motor_signals(self):
        """Создать реалистичные сигналы двигателя"""
        sample_rate = 25600
        duration = 0.1  # 100 мс
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Имитируем токи трехфазного двигателя
        fundamental_freq = 50  # Основная частота 50 Гц

        phase_a = 10 * np.sin(2 * np.pi * fundamental_freq * t) + \
                  0.5 * np.sin(2 * np.pi * 150 * t) + \  # 3-я гармоника
                  0.1 * np.random.normal(0, 1, len(t))     # Шум

        phase_b = 10 * np.sin(2 * np.pi * fundamental_freq * t + 2*np.pi/3) + \
                  0.5 * np.sin(2 * np.pi * 150 * t + 2*np.pi/3) + \
                  0.1 * np.random.normal(0, 1, len(t))

        phase_c = 10 * np.sin(2 * np.pi * fundamental_freq * t + 4*np.pi/3) + \
                  0.5 * np.sin(2 * np.pi * 150 * t + 4*np.pi/3) + \
                  0.1 * np.random.normal(0, 1, len(t))

        return phase_a, phase_b, phase_c

    def test_realistic_motor_signal_processing(self, sample_motor_signals):
        """Тест обработки реалистичных сигналов двигателя"""
        phase_a, phase_b, phase_c = sample_motor_signals

        extractor = FeatureExtractor(sample_rate=25600)

        features = extractor.extract_features_from_phases(
            phase_a=phase_a,
            phase_b=phase_b,
            phase_c=phase_c
        )

        # Все три фазы должны быть обработаны
        assert features['summary']['processed_phases'] == 3

        # Проверяем, что найдена основная частота около 50 Гц
        for phase_key in ['a', 'b', 'c']:
            phase_features = features['phases'][phase_key]
            dominant_freq = phase_features['frequency']['dominant_frequency']

            # Доминантная частота должна быть около 50 Гц
            assert 40 <= dominant_freq <= 60

            # RMS должен быть разумным для амплитуды ~10
            rms = phase_features['statistical']['rms']
            assert 6 <= rms <= 15  # Учитываем шум и гармоники


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
