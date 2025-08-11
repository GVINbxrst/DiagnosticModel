"""
Модуль извлечения признаков из токовых сигналов асинхронных двигателей

Этот модуль выполняет цифровую обработку сигналов для извлечения диагностических признаков:
- Интерполяция пропусков и очистка NaN значений
- Расчет статистических характеристик (RMS, crest factor, skewness, kurtosis)
- FFT анализ с оконной функцией Hann
- Поиск пиков спектра и их характеристик
- Сохранение результатов в базу данных PostgreSQL
"""

import asyncio
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, windows
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import Feature, RawSignal
from src.utils.logger import get_logger

# Настройки
settings = get_settings()
logger = get_logger(__name__)

# Константы для обработки сигналов
DEFAULT_SAMPLE_RATE = 25600  # Гц
DEFAULT_WINDOW_SIZE = 4096   # Размер окна FFT согласно контракту
MIN_SIGNAL_LENGTH = 100      # Минимальная длина сигнала для анализа
MAX_NAN_RATIO = 0.2         # Порог доли NaN: >20% -> фаза отсутствует (не обрабатываем)


class SignalProcessingError(Exception):
    """Базовое исключение для обработки сигналов"""
    pass


class InsufficientDataError(SignalProcessingError):
    """Исключение для недостаточного количества данных"""
    pass


class SignalPreprocessor:
    """Предобработчик сигналов для очистки и интерполяции"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def clean_and_interpolate_signal(
        self,
        signal_data: np.ndarray,
        max_nan_ratio: float = MAX_NAN_RATIO
    ) -> np.ndarray:
        """
        Очистить сигнал от NaN и интерполировать пропуски

        Args:
            signal_data: Исходный сигнал с возможными NaN
            max_nan_ratio: Максимальная доля NaN значений

        Returns:
            Очищенный сигнал без NaN

        Raises:
            InsufficientDataError: Если слишком много NaN значений
        """
        if len(signal_data) == 0:
            raise InsufficientDataError("Пустой сигнал")

        # Проверяем долю NaN значений
        nan_mask = np.isnan(signal_data)
        nan_ratio = np.sum(nan_mask) / len(signal_data)

        if nan_ratio > max_nan_ratio:
            raise InsufficientDataError(
                f"Превышен порог NaN {nan_ratio:.1%} > {max_nan_ratio:.1%}"
            )

        # Если нет NaN, возвращаем как есть
        if nan_ratio == 0:
            return signal_data.copy()

        # Обрезаем NaN в начале и конце
        signal_trimmed = self._trim_leading_trailing_nan(signal_data)

        if len(signal_trimmed) < MIN_SIGNAL_LENGTH:
            raise InsufficientDataError(
                f"Слишком короткий сигнал после обрезки NaN: {len(signal_trimmed)}"
            )

        # Интерполируем внутренние пропуски
        signal_interpolated = self._interpolate_internal_nan(signal_trimmed)

        # Медианный фильтр (окно 51) при наличии шума
        if signal_interpolated.size >= 51:
            from scipy.signal import medfilt
            try:
                filtered = medfilt(signal_interpolated, kernel_size=51)
                signal_interpolated = filtered
            except Exception:  # тихо игнорируем сбой фильтра, оставляем необработанным
                self.logger.debug("Не удалось применить медианный фильтр", exc_info=True)

        self.logger.debug(
            f"Обработка сигнала: {len(signal_data)} -> {len(signal_interpolated)} отсчетов, "
            f"NaN: {nan_ratio:.1%}"
        )

        return signal_interpolated

    def _trim_leading_trailing_nan(self, signal_data: np.ndarray) -> np.ndarray:
        """Обрезать NaN в начале и конце сигнала"""
        # Находим первый и последний не-NaN элементы
        valid_indices = np.where(~np.isnan(signal_data))[0]

        if len(valid_indices) == 0:
            raise InsufficientDataError("Сигнал состоит только из NaN")

        start_idx = valid_indices[0]
        end_idx = valid_indices[-1] + 1

        return signal_data[start_idx:end_idx]

    def _interpolate_internal_nan(self, signal_data: np.ndarray) -> np.ndarray:
        """Интерполировать внутренние NaN значения"""
        nan_mask = np.isnan(signal_data)

        # Если нет NaN, возвращаем как есть
        if not np.any(nan_mask):
            return signal_data.copy()

        # Создаем индексы
        indices = np.arange(len(signal_data))
        valid_indices = indices[~nan_mask]
        valid_values = signal_data[~nan_mask]

        # Линейная интерполяция
        interpolator = interp1d(
            valid_indices,
            valid_values,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        # Заполняем NaN значения
        result = signal_data.copy()
        result[nan_mask] = interpolator(indices[nan_mask])

        return result


class StatisticalFeatureExtractor:
    """Извлечение статистических признаков из сигналов"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def extract_statistical_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Извлечь статистические признаки из сигнала

        Args:
            signal_data: Очищенный сигнал

        Returns:
            Словарь с статистическими признаками
        """
        if len(signal_data) == 0:
            raise ValueError("Пустой сигнал")

        # Базовая статистика
        mean_val = float(np.mean(signal_data))
        std_val = float(np.std(signal_data))
        min_val = float(np.min(signal_data))
        max_val = float(np.max(signal_data))

        # RMS (Root Mean Square)
        rms = float(np.sqrt(np.mean(signal_data**2)))

        # Crest Factor (отношение пикового значения к RMS)
        peak_val = max(abs(min_val), abs(max_val))
        crest_factor = peak_val / rms if rms > 0 else 0.0

        # Skewness (асимметрия)
        skewness = float(stats.skew(signal_data))

        # Kurtosis (эксцесс)
        kurtosis = float(stats.kurtosis(signal_data))

        features = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'rms': rms,
            'crest_factor': crest_factor,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

        self.logger.debug(f"Статистические признаки: RMS={rms:.4f}, Crest={crest_factor:.2f}")

        return features


class FrequencyFeatureExtractor:
    """Извлечение частотных признаков с помощью FFT"""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def extract_fft_features(
        self,
        signal_data: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    top_peaks: int = 10
    ) -> Dict:
        """
        Извлечь частотные признаки с помощью FFT

        Args:
            signal_data: Очищенный сигнал
            window_size: Размер окна FFT
            top_peaks: Количество главных пиков для извлечения

        Returns:
            Словарь с частотными характеристиками
        """
        if len(signal_data) < window_size:
            # Если сигнал короче окна, используем его длину
            window_size = len(signal_data)
            self.logger.warning(f"Сигнал короче окна FFT, используем {window_size} отсчетов")

        # Применяем оконную функцию Hann
        window = windows.hann(window_size)

        # Разбиваем сигнал на перекрывающиеся окна
        hop_size = window_size // 2  # 50% перекрытие
        windowed_segments = self._create_windowed_segments(signal_data, window_size, hop_size)

        if len(windowed_segments) == 0:
            raise InsufficientDataError("Недостаточно данных для FFT анализа")

        # Вычисляем FFT для каждого сегмента и усредняем
        magnitude_spectrum = self._compute_average_spectrum(windowed_segments, window)

        # Создаем массив частот
        frequencies = np.fft.fftfreq(window_size, 1/self.sample_rate)[:window_size//2]

        # Находим пики
        peaks_info = self._find_spectral_peaks(frequencies, magnitude_spectrum, top_peaks)

        # Вычисляем дополнительные частотные характеристики
        spectral_features = self._compute_spectral_features(frequencies, magnitude_spectrum)

        fft_features = {
            'frequencies': frequencies.tolist(),
            'magnitude_spectrum': magnitude_spectrum.tolist(),
            'peaks': peaks_info,
            'spectral_centroid': spectral_features['centroid'],
            'spectral_bandwidth': spectral_features['bandwidth'],
            'spectral_rolloff': spectral_features['rolloff'],
            'spectral_energy': spectral_features['energy'],
            'dominant_frequency': peaks_info[0]['frequency'] if peaks_info else 0.0,
            'window_size': window_size,
            'sample_rate': self.sample_rate
        }

        self.logger.debug(
            f"FFT анализ: {len(peaks_info)} пиков, "
            f"доминантная частота: {fft_features['dominant_frequency']:.1f} Гц"
        )

        return fft_features

    def _create_windowed_segments(
        self,
        signal_data: np.ndarray,
        window_size: int,
        hop_size: int
    ) -> List[np.ndarray]:
        """Создать перекрывающиеся сегменты сигнала"""
        segments = []

        for start in range(0, len(signal_data) - window_size + 1, hop_size):
            segment = signal_data[start:start + window_size]
            segments.append(segment)

        return segments

    def _compute_average_spectrum(
        self,
        segments: List[np.ndarray],
        window: np.ndarray
    ) -> np.ndarray:
        """Вычислить усредненный спектр по всем сегментам"""
        magnitude_spectra = []

        for segment in segments:
            # Применяем окно
            windowed_segment = segment * window

            # Вычисляем FFT
            fft_result = np.fft.fft(windowed_segment)

            # Берем только положительные частоты и вычисляем магнитуду
            magnitude_spectrum = np.abs(fft_result[:len(fft_result)//2])
            magnitude_spectra.append(magnitude_spectrum)

        # Усредняем по всем сегментам
        average_spectrum = np.mean(magnitude_spectra, axis=0)

        return average_spectrum

    def _find_spectral_peaks(
        self,
        frequencies: np.ndarray,
        magnitude_spectrum: np.ndarray,
        top_peaks: int
    ) -> List[Dict]:
        """Найти главные пики в спектре"""
        # Параметры поиска пиков
        # Минимальная высота пика (10% от максимума)
        min_height = 0.1 * np.max(magnitude_spectrum)

        # Минимальное расстояние между пиками (50 Гц)
        min_distance = int(50 * len(frequencies) / (self.sample_rate / 2))

        # Находим пики
        peak_indices, peak_properties = find_peaks(
            magnitude_spectrum,
            height=min_height,
            distance=min_distance,
            prominence=min_height * 0.5
        )

        # Сортируем по амплитуде (убывание)
        peak_amplitudes = magnitude_spectrum[peak_indices]
        sorted_indices = np.argsort(peak_amplitudes)[::-1]

        # Берем топ пиков
        top_indices = sorted_indices[:min(top_peaks, len(sorted_indices))]

        peaks_info = []
        for idx in top_indices:
            peak_idx = peak_indices[idx]
            peaks_info.append({
                'frequency': float(frequencies[peak_idx]),
                'amplitude': float(magnitude_spectrum[peak_idx]),
                'prominence': float(peak_properties['prominences'][idx]),
                'width': float(peak_properties.get('widths', [0])[idx] if 'widths' in peak_properties else 0)
            })

        return peaks_info

    def _compute_spectral_features(
        self,
        frequencies: np.ndarray,
        magnitude_spectrum: np.ndarray
    ) -> Dict[str, float]:
        """Вычислить дополнительные спектральные характеристики"""
        # Нормализуем спектр для вычисления центроида
        total_energy = np.sum(magnitude_spectrum)

        if total_energy == 0:
            return {
                'centroid': 0.0,
                'bandwidth': 0.0,
                'rolloff': 0.0,
                'energy': 0.0
            }

        normalized_spectrum = magnitude_spectrum / total_energy

        # Спектральный центроид (центр масс спектра)
        centroid = np.sum(frequencies * normalized_spectrum)

        # Спектральная ширина (стандартное отклонение от центроида)
        bandwidth = np.sqrt(np.sum(((frequencies - centroid) ** 2) * normalized_spectrum))

        # Спектральный rolloff (частота, ниже которой 85% энергии)
        cumulative_energy = np.cumsum(normalized_spectrum)
        rolloff_threshold = 0.85
        rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
        rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]

        return {
            'centroid': float(centroid),
            'bandwidth': float(bandwidth),
            'rolloff': float(rolloff),
            'energy': float(total_energy)
        }


class FeatureExtractor:
    """Основной класс для извлечения признаков из токовых сигналов"""

    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.preprocessor = SignalPreprocessor()
        self.statistical_extractor = StatisticalFeatureExtractor()
        self.frequency_extractor = FrequencyFeatureExtractor(sample_rate)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def extract_features_from_phases(
        self,
        phase_a: Optional[np.ndarray] = None,
        phase_b: Optional[np.ndarray] = None,
        phase_c: Optional[np.ndarray] = None,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None
    ) -> Dict:
        """
        Извлечь признаки из трех фаз тока

        Args:
            phase_a: Сигнал фазы A (R)
            phase_b: Сигнал фазы B (S)
            phase_c: Сигнал фазы C (T)
            window_start: Начало временного окна
            window_end: Конец временного окна

        Returns:
            Словарь с извлеченными признаками
        """
        features = {
            'window_info': {
                'start': window_start.isoformat() if window_start else None,
                'end': window_end.isoformat() if window_end else None,
                'sample_rate': self.sample_rate
            },
            'phases': {}
        }

        phases = {'a': phase_a, 'b': phase_b, 'c': phase_c}
        phase_names = {'a': 'R', 'b': 'S', 'c': 'T'}

        processed_phases = 0

        for phase_key, phase_data in phases.items():
            phase_name = phase_names[phase_key]

            if phase_data is None or len(phase_data) == 0:
                self.logger.debug(f"Фаза {phase_name}: нет данных")
                features['phases'][phase_key] = None
                continue

            try:
                # Предобработка сигнала
                clean_signal = self.preprocessor.clean_and_interpolate_signal(phase_data)

                # Извлечение статистических признаков
                statistical_features = self.statistical_extractor.extract_statistical_features(clean_signal)

                # Извлечение частотных признаков
                fft_features = self.frequency_extractor.extract_fft_features(clean_signal)

                # Объединяем все признаки для фазы
                phase_features = {
                    'statistical': statistical_features,
                    'frequency': fft_features,
                    'data_quality': {
                        'original_length': len(phase_data),
                        'processed_length': len(clean_signal),
                        'nan_ratio': float(np.sum(np.isnan(phase_data)) / len(phase_data))
                    }
                }

                features['phases'][phase_key] = phase_features
                processed_phases += 1

                self.logger.debug(
                    f"Фаза {phase_name}: обработано {len(clean_signal)} отсчетов, "
                    f"RMS={statistical_features['rms']:.4f}"
                )

            except Exception as e:
                self.logger.warning(f"Ошибка обработки фазы {phase_name}: {e}")
                features['phases'][phase_key] = None

        if processed_phases == 0:
            raise InsufficientDataError("Ни одна фаза не была успешно обработана")

        # Добавляем сводную информацию
        features['summary'] = {
            'processed_phases': processed_phases,
            'total_phases': len([p for p in phases.values() if p is not None]),
            'processing_timestamp': datetime.utcnow().isoformat()
        }

        self.logger.info(f"Извлечены признаки из {processed_phases} фаз")

        return features

    async def process_raw_signal(
        self,
        raw_signal_id: UUID,
        window_duration_ms: int = 1000,
        overlap_ratio: float = 0.5
    ) -> List[UUID]:
        """
        Обработать сырой сигнал и сохранить признаки в базу данных

        Args:
            raw_signal_id: ID сырого сигнала
            window_duration_ms: Длительность окна в миллисекундах
            overlap_ratio: Коэффициент перекрытия окон (0.0 - 1.0)

        Returns:
            Список ID созданных записей признаков
        """
        async with get_async_session() as session:
            # Загружаем сырой сигнал
            raw_signal = await session.get(RawSignal, raw_signal_id)
            if not raw_signal:
                raise ValueError(f"Сырой сигнал с ID {raw_signal_id} не найден")

            # Распаковываем данные фаз
                        from src.utils.serialization import load_float32_array

                        phase_a = load_float32_array(raw_signal.phase_a) if raw_signal.phase_a else None
                        phase_b = load_float32_array(raw_signal.phase_b) if raw_signal.phase_b else None
                        phase_c = load_float32_array(raw_signal.phase_c) if raw_signal.phase_c else None

            # Вычисляем параметры окон
            samples_per_window = int(window_duration_ms * self.sample_rate / 1000)
            # Принудительно приводим окно FFT к 4096 если возможно
            if samples_per_window < 4096:
                samples_per_window = 4096 if max(len(x) if x is not None else 0 for x in [phase_a, phase_b, phase_c]) >= 4096 else samples_per_window
            hop_size = int(samples_per_window * (1 - overlap_ratio))

            # Определяем максимальную длину сигнала
            max_length = 0
            for phase in [phase_a, phase_b, phase_c]:
                if phase is not None:
                    max_length = max(max_length, len(phase))

            if max_length < samples_per_window:
                # Обрабатываем весь сигнал как одно окно
                features = self.extract_features_from_phases(
                    phase_a, phase_b, phase_c,
                    raw_signal.recorded_at,
                    raw_signal.recorded_at + timedelta(milliseconds=window_duration_ms)
                )

                feature_id = await self._save_features_to_db(
                    session, raw_signal_id, features,
                    raw_signal.recorded_at,
                    raw_signal.recorded_at + timedelta(milliseconds=window_duration_ms)
                )

                await session.commit()
                return [feature_id]

            # Обрабатываем по окнам
            feature_ids = []

            for start_sample in range(0, max_length - samples_per_window + 1, hop_size):
                end_sample = start_sample + samples_per_window

                # Извлекаем сегменты для каждой фазы
                segment_a = phase_a[start_sample:end_sample] if phase_a is not None else None
                segment_b = phase_b[start_sample:end_sample] if phase_b is not None else None
                segment_c = phase_c[start_sample:end_sample] if phase_c is not None else None

                # Вычисляем временные метки
                start_time = raw_signal.recorded_at + timedelta(
                    seconds=start_sample / self.sample_rate
                )
                end_time = raw_signal.recorded_at + timedelta(
                    seconds=end_sample / self.sample_rate
                )

                try:
                    # Извлекаем признаки
                    features = self.extract_features_from_phases(
                        segment_a, segment_b, segment_c, start_time, end_time
                    )

                    # Сохраняем в базу данных
                    feature_id = await self._save_features_to_db(
                        session, raw_signal_id, features, start_time, end_time
                    )

                    feature_ids.append(feature_id)

                except InsufficientDataError as e:
                    self.logger.warning(f"Пропускаем окно {start_sample}-{end_sample}: {e}")
                    continue

            # Помечаем сырой сигнал как обработанный
            raw_signal.processed = True

            await session.commit()

            self.logger.info(
                f"Обработан сигнал {raw_signal_id}: создано {len(feature_ids)} записей признаков"
            )

            return feature_ids

    async def _save_features_to_db(
        self,
        session: AsyncSession,
        raw_signal_id: UUID,
        features: Dict,
        window_start: datetime,
        window_end: datetime
    ) -> UUID:
        """Сохранить извлеченные признаки в базу данных"""

        # Извлекаем признаки для каждой фазы
        def get_phase_feature(phase_key: str, feature_type: str, feature_name: str) -> Optional[float]:
            phase_data = features['phases'].get(phase_key)
            if phase_data and phase_data.get(feature_type):
                return phase_data[feature_type].get(feature_name)
            return None

        # Создаем объект признаков
        feature_record = Feature(
            raw_id=raw_signal_id,
            window_start=window_start,
            window_end=window_end,

            # Статистические признаки фазы A (R)
            rms_a=get_phase_feature('a', 'statistical', 'rms'),
            crest_a=get_phase_feature('a', 'statistical', 'crest_factor'),
            kurt_a=get_phase_feature('a', 'statistical', 'kurtosis'),
            skew_a=get_phase_feature('a', 'statistical', 'skewness'),
            mean_a=get_phase_feature('a', 'statistical', 'mean'),
            std_a=get_phase_feature('a', 'statistical', 'std'),
            min_a=get_phase_feature('a', 'statistical', 'min'),
            max_a=get_phase_feature('a', 'statistical', 'max'),

            # Статистические признаки фазы B (S)
            rms_b=get_phase_feature('b', 'statistical', 'rms'),
            crest_b=get_phase_feature('b', 'statistical', 'crest_factor'),
            kurt_b=get_phase_feature('b', 'statistical', 'kurtosis'),
            skew_b=get_phase_feature('b', 'statistical', 'skewness'),
            mean_b=get_phase_feature('b', 'statistical', 'mean'),
            std_b=get_phase_feature('b', 'statistical', 'std'),
            min_b=get_phase_feature('b', 'statistical', 'min'),
            max_b=get_phase_feature('b', 'statistical', 'max'),

            # Статистические признаки фазы C (T)
            rms_c=get_phase_feature('c', 'statistical', 'rms'),
            crest_c=get_phase_feature('c', 'statistical', 'crest_factor'),
            kurt_c=get_phase_feature('c', 'statistical', 'kurtosis'),
            skew_c=get_phase_feature('c', 'statistical', 'skewness'),
            mean_c=get_phase_feature('c', 'statistical', 'mean'),
            std_c=get_phase_feature('c', 'statistical', 'std'),
            min_c=get_phase_feature('c', 'statistical', 'min'),
            max_c=get_phase_feature('c', 'statistical', 'max'),

            # FFT спектр и дополнительные признаки
            fft_spectrum=self._prepare_fft_spectrum_for_db(features),
            extra=features
        )

        session.add(feature_record)
        await session.flush()

        return feature_record.id

    def _prepare_fft_spectrum_for_db(self, features: Dict) -> Dict:
        """Подготовить FFT спектр для сохранения в JSONB"""
        fft_data = {}

        for phase_key in ['a', 'b', 'c']:
            phase_data = features['phases'].get(phase_key)
            if phase_data and phase_data.get('frequency'):
                freq_data = phase_data['frequency']

                # Сохраняем только ключевую информацию для экономии места
                fft_data[f'phase_{phase_key}'] = {
                    'peaks': freq_data.get('peaks', []),
                    'spectral_centroid': freq_data.get('spectral_centroid'),
                    'spectral_bandwidth': freq_data.get('spectral_bandwidth'),
                    'spectral_rolloff': freq_data.get('spectral_rolloff'),
                    'spectral_energy': freq_data.get('spectral_energy'),
                    'dominant_frequency': freq_data.get('dominant_frequency'),
                    'sample_rate': freq_data.get('sample_rate')
                }

        return fft_data


# Вспомогательные функции для CLI использования

async def process_unprocessed_signals(
    limit: Optional[int] = None,
    window_duration_ms: int = 1000,
    overlap_ratio: float = 0.5
) -> Dict[str, int]:
    """
    Обработать все необработанные сигналы в базе данных

    Args:
        limit: Максимальное количество сигналов для обработки
        window_duration_ms: Длительность окна в миллисекундах
        overlap_ratio: Коэффициент перекрытия окон

    Returns:
        Статистика обработки
    """
    from sqlalchemy import select

    stats = {
        'processed_signals': 0,
        'created_features': 0,
        'errors': 0
    }

    feature_extractor = FeatureExtractor()

    async with get_async_session() as session:
        # Получаем необработанные сигналы
        query = select(RawSignal).where(RawSignal.processed == False)
        if limit:
            query = query.limit(limit)

        result = await session.execute(query)
        unprocessed_signals = result.scalars().all()

        logger.info(f"Найдено {len(unprocessed_signals)} необработанных сигналов")

        for raw_signal in unprocessed_signals:
            try:
                feature_ids = await feature_extractor.process_raw_signal(
                    raw_signal.id,
                    window_duration_ms=window_duration_ms,
                    overlap_ratio=overlap_ratio
                )

                stats['processed_signals'] += 1
                stats['created_features'] += len(feature_ids)

                logger.info(
                    f"Обработан сигнал {raw_signal.id}: создано {len(feature_ids)} признаков"
                )

            except Exception as e:
                logger.error(f"Ошибка обработки сигнала {raw_signal.id}: {e}")
                stats['errors'] += 1

    return stats


if __name__ == "__main__":
    # Пример использования
    import argparse

    parser = argparse.ArgumentParser(description="Извлечение признаков из токовых сигналов")
    parser.add_argument("--limit", type=int, help="Максимальное количество сигналов для обработки")
    parser.add_argument("--window-ms", type=int, default=1000, help="Длительность окна в мс")
    parser.add_argument("--overlap", type=float, default=0.5, help="Коэффициент перекрытия (0.0-1.0)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                       help="Частота дискретизации (Гц)")

    args = parser.parse_args()

    async def main():
        logger.info("Начинаем извлечение признаков из токовых сигналов")

        stats = await process_unprocessed_signals(
            limit=args.limit,
            window_duration_ms=args.window_ms,
            overlap_ratio=args.overlap
        )

        print(f"\n✓ Обработка завершена:")
        print(f"  - Обработано сигналов: {stats['processed_signals']}")
        print(f"  - Создано признаков: {stats['created_features']}")
        print(f"  - Ошибок: {stats['errors']}")

    asyncio.run(main())
