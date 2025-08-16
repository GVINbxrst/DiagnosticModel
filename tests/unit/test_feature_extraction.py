"""Восстановленные unit и интеграционные тесты для feature_extraction.
Тесты согласованы с бизнес-правилами: MIN_SIGNAL_LENGTH=100 применяется только
к сигналам содержащим NaN (иначе короткие чистые сигналы допускаются),
MAX_NAN_RATIO=0.2. Покрытие: предобработка, статистика, FFT, агрегатор, БД save, интеграция.
"""

from datetime import datetime, timedelta, UTC
from uuid import uuid4
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.data_processing.feature_extraction import (
    SignalPreprocessor,
    StatisticalFeatureExtractor,
    FrequencyFeatureExtractor,
    FeatureExtractor,
    InsufficientDataError,
)


class TestSignalPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return SignalPreprocessor()

    def test_clean_signal_without_nan_short_allowed(self, preprocessor):
        # Короткий сигнал без NaN возвращается как есть
        sig = np.array([1.0, 2.0, 3.0])
        cleaned = preprocessor.clean_and_interpolate_signal(sig)
        np.testing.assert_array_equal(cleaned, sig)

    def test_interpolate_internal_nan_long(self, preprocessor):
        sig = np.linspace(0, 10, 150)
        sig[70] = np.nan
        cleaned = preprocessor.clean_and_interpolate_signal(sig)
        assert not np.isnan(cleaned).any()
        assert abs(cleaned[70] - (cleaned[69] + cleaned[71]) / 2) < 1e-6

    def test_trim_edges_nan_long(self, preprocessor):
        inner = np.linspace(-5, 5, 150)
        sig = np.concatenate(([np.nan] * 5, inner, [np.nan] * 3))
        cleaned = preprocessor.clean_and_interpolate_signal(sig)
        assert len(cleaned) == len(inner)
        assert not np.isnan(cleaned).any()

    def test_too_many_nan(self, preprocessor):
        sig = np.ones(150)
        # 33% NaN > 20%
        sig[::3] = np.nan
        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(sig)

    def test_all_nan(self, preprocessor):
        sig = np.array([np.nan] * 150)
        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(sig)

    def test_empty(self, preprocessor):
        with pytest.raises(InsufficientDataError):
            preprocessor.clean_and_interpolate_signal(np.array([]))


class TestStatisticalFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return StatisticalFeatureExtractor()

    def test_basic_stats(self, extractor):
        sig = np.array([1, 2, 3, 4, 5], dtype=float)
        f = extractor.extract_statistical_features(sig)
        assert f['mean'] == 3.0 and f['min'] == 1.0 and f['max'] == 5.0
        expected_rms = np.sqrt(np.mean(sig ** 2))
        assert abs(f['rms'] - expected_rms) < 1e-9
        crest = 5.0 / expected_rms
        assert abs(f['crest_factor'] - crest) < 1e-9

    def test_symmetry(self, extractor):
        sig = np.array([-2, -1, 0, 1, 2], dtype=float)
        f = extractor.extract_statistical_features(sig)
        assert abs(f['skewness']) < 1e-10

    def test_kurtosis(self, extractor):
        np.random.seed(42)
        sig = np.random.normal(0, 1, 8000)
        f = extractor.extract_statistical_features(sig)
        assert abs(f['kurtosis']) < 0.25

    def test_zero(self, extractor):
        sig = np.zeros(128)
        f = extractor.extract_statistical_features(sig)
        assert f['rms'] == 0.0 and f['crest_factor'] == 0.0

    def test_single_value(self, extractor):
        sig = np.array([5.0])
        f = extractor.extract_statistical_features(sig)
        assert f['std'] == 0.0 and f['mean'] == 5.0


class TestFrequencyFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FrequencyFeatureExtractor(sample_rate=1000)

    def test_fft_sine(self, extractor):
        sr = 1000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t)
        f = extractor.extract_fft_features(sig, window_size=512)
        assert abs(f['dominant_frequency'] - 50) < 5
        assert len(f['peaks']) > 0

    def test_multiple_components(self, extractor):
        sr = 1000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        f = extractor.extract_fft_features(sig, window_size=512, top_peaks=5)
        assert len(f['peaks']) >= 2

    def test_top_peaks_limit(self, extractor):
        sr = 1000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sig = sum(np.sin(2 * np.pi * f * t) for f in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        f = extractor.extract_fft_features(sig, window_size=512, top_peaks=10)
        assert len(f['peaks']) <= 10

    def test_window_auto_reduce(self, extractor):
        sr = 1000
        t = np.linspace(0, 0.15, int(sr * 0.15), endpoint=False)
        sig = np.sin(2 * np.pi * 60 * t)
        f = extractor.extract_fft_features(sig, window_size=4096)
        assert f['window_size'] == len(sig)

    def test_spectral_metrics(self, extractor):
        np.random.seed(7)
        sig = np.random.normal(0, 1, 1024)
        f = extractor.extract_fft_features(sig)
        for k in ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'spectral_energy']:
            assert f[k] > 0


class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(sample_rate=1000)

    def test_single_phase(self, extractor):
        t = np.linspace(0, 1, 1500)
        phase_a = np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.normal(0, 1, len(t))
        ws = datetime.now(UTC); we = ws + timedelta(seconds=1)
        f = extractor.extract_features_from_phases(phase_a=phase_a, window_start=ws, window_end=we)
        assert f['phases']['a'] and f['phases']['b'] is None and f['phases']['c'] is None
        assert 'statistical' in f['phases']['a'] and 'frequency' in f['phases']['a']

    def test_three_phases(self, extractor):
        t = np.linspace(0, 1, 1500)
        a = np.sin(2 * np.pi * 50 * t)
        b = np.sin(2 * np.pi * 50 * t + 2 * np.pi / 3)
        c = np.sin(2 * np.pi * 50 * t + 4 * np.pi / 3)
        f = extractor.extract_features_from_phases(phase_a=a, phase_b=b, phase_c=c)
        assert f['summary']['processed_phases'] == 3

    def test_with_nan(self, extractor):
        t = np.linspace(0, 1, 1500)
        a = np.sin(2 * np.pi * 50 * t)
        a[600:650] = np.nan
        f = extractor.extract_features_from_phases(phase_a=a)
        dq = f['phases']['a']['data_quality']
        assert dq['original_length'] == 1500 and dq['processed_length'] <= 1500 and dq['nan_ratio'] > 0

    def test_short_signal_with_nan_error(self, extractor):
        # Содержит NaN и слишком короткий после тримминга -> ошибка
        with pytest.raises(InsufficientDataError):
            extractor.extract_features_from_phases(phase_a=np.array([1.0, np.nan]))

    def test_all_none(self, extractor):
        with pytest.raises(InsufficientDataError):
            extractor.extract_features_from_phases(phase_a=None, phase_b=None, phase_c=None)

    @pytest.mark.asyncio
    async def test_save_features_to_db(self, extractor):
        class DummySession:
            def __init__(self):
                from unittest.mock import MagicMock, AsyncMock as _AsyncMock
                self.add = MagicMock()
                self.flush = _AsyncMock()
            # совместимость если код вдруг вызовет commit
            async def commit(self):
                return None
        mock_session = DummySession()
        features = {
            'phases': {
                'a': {
                    'statistical': {
                        'rms': 1.5, 'crest_factor': 1.2, 'kurtosis': 0.1,
                        'skewness': 0.05, 'mean': 1.0, 'std': 0.5, 'min': 0.0, 'max': 2.0
                    },
                    'frequency': {
                        'peaks': [{'frequency': 50.0, 'amplitude': 1.0}],
                        'spectral_centroid': 75.0,
                        'spectral_bandwidth': 10.0,
                        'spectral_rolloff': 120.0,
                        'spectral_energy': 123.0,
                        'dominant_frequency': 50.0,
                        'sample_rate': 1000,
                        'window_size': 128
                    }
                },
                'b': None,
                'c': None
            }
        }
        rid = uuid4(); ws = datetime.now(UTC); we = ws + timedelta(seconds=1)
        _ = await extractor._save_features_to_db(mock_session, rid, features, ws, we)
        assert mock_session.add.called and mock_session.flush.await_count == 1


@pytest.mark.integration
class TestFeatureExtractionIntegration:
    @pytest.fixture
    def sample_motor_signals(self):
        sr = 25600; duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        base = 10; fund = 50
        a = base * np.sin(2 * np.pi * fund * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.normal(0, 1, len(t))
        b = base * np.sin(2 * np.pi * fund * t + 2 * np.pi / 3) + 0.5 * np.sin(2 * np.pi * 150 * t + 2 * np.pi / 3) + 0.1 * np.random.normal(0, 1, len(t))
        c = base * np.sin(2 * np.pi * fund * t + 4 * np.pi / 3) + 0.5 * np.sin(2 * np.pi * 150 * t + 4 * np.pi / 3) + 0.1 * np.random.normal(0, 1, len(t))
        return a, b, c

    def test_realistic(self, sample_motor_signals):
        a, b, c = sample_motor_signals
        extractor = FeatureExtractor(sample_rate=25600)
        f = extractor.extract_features_from_phases(phase_a=a, phase_b=b, phase_c=c)
        assert f['summary']['processed_phases'] == 3
        for ph in ['a', 'b', 'c']:
            dom = f['phases'][ph]['frequency']['dominant_frequency']
            assert 40 <= dom <= 60
            rms = f['phases'][ph]['statistical']['rms']
            assert 6 <= rms <= 15


if __name__ == '__main__':  # pragma: no cover
    pytest.main([__file__, '-q'])
