"""
Утилиты для Dashboard приложения
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import hashlib
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class SessionManager:
    """Менеджер сессий для Dashboard"""

    @staticmethod
    def init_session_state():
        """Инициализация состояния сессии"""
        if 'token' not in st.session_state:
            st.session_state.token = None

        if 'user_info' not in st.session_state:
            st.session_state.user_info = {}

        if 'selected_equipment' not in st.session_state:
            st.session_state.selected_equipment = None

        if 'dashboard_settings' not in st.session_state:
            st.session_state.dashboard_settings = {
                'auto_refresh': False,
                'refresh_interval': 60,
                'chart_theme': 'plotly_white',
                'show_tooltips': True
            }

    @staticmethod
    def clear_session():
        """Очистка сессии"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    @staticmethod
    def is_authenticated() -> bool:
        """Проверка аутентификации"""
        return st.session_state.get('token') is not None

    @staticmethod
    def get_user_role() -> str:
        """Получение роли пользователя"""
        return st.session_state.get('user_info', {}).get('role', 'guest')

class DataCache:
    """Кеширование данных для улучшения производительности"""

    @staticmethod
    @st.cache_data(ttl=300)  # Кеш на 5 минут
    def get_equipment_list(api_url: str, headers: Dict) -> List[Dict]:
        """Кешированное получение списка оборудования"""
        import requests
        try:
            response = requests.get(f"{api_url}/equipment", headers=headers)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения оборудования: {e}")
            return []

    @staticmethod
    @st.cache_data(ttl=60)  # Кеш на 1 минуту
    def get_equipment_files(api_url: str, headers: Dict, equipment_id: int) -> List[Dict]:
        """Кешированное получение файлов оборудования"""
        import requests
        try:
            response = requests.get(
                f"{api_url}/equipment/{equipment_id}/files",
                headers=headers
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error(f"Ошибка получения файлов: {e}")
            return []

    @staticmethod
    @st.cache_data(ttl=300)  # Кеш на 5 минут
    def get_signal_data(api_url: str, headers: Dict, raw_id: int) -> Optional[Dict]:
        """Кешированное получение данных сигнала"""
        import requests
        try:
            response = requests.get(f"{api_url}/signals/{raw_id}", headers=headers)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Ошибка получения сигнала: {e}")
            return None

    @staticmethod
    def clear_cache():
        """Очистка всех кешей"""
        st.cache_data.clear()

class DataProcessor:
    """Обработка данных для визуализации"""

    @staticmethod
    def prepare_time_series_data(signal_data: Dict) -> pd.DataFrame:
        """Подготовка данных временных рядов"""
        if not signal_data:
            return pd.DataFrame()

        sample_rate = signal_data.get('sample_rate_hz', 25600)
        samples_count = signal_data.get('samples_count', 0)

        if samples_count == 0:
            return pd.DataFrame()

        # Создание временной оси
        time_axis = np.linspace(0, samples_count / sample_rate, samples_count)

        # Подготовка данных фаз
        data = {'time': time_axis}

        phases = ['phase_a', 'phase_b', 'phase_c']
        phase_names = ['R', 'S', 'T']

        for phase, name in zip(phases, phase_names):
            if phase in signal_data and signal_data[phase]:
                phase_data = np.array(signal_data[phase])
                # Обрезаем до минимальной длины
                min_length = min(len(time_axis), len(phase_data))
                data[f'Current_{name}'] = phase_data[:min_length]

        df = pd.DataFrame(data)
        return df

    @staticmethod
    def prepare_fft_data(features: Dict) -> pd.DataFrame:
        """Подготовка данных FFT"""
        if not features or 'fft_spectrum' not in features:
            return pd.DataFrame()

        fft_data = features['fft_spectrum']
        combined_data = []

        phases = ['phase_a', 'phase_b', 'phase_c']
        phase_names = ['R', 'S', 'T']

        for phase, name in zip(phases, phase_names):
            if phase in fft_data:
                spectrum = fft_data[phase]
                frequencies = spectrum.get('frequencies', [])
                magnitudes = spectrum.get('magnitudes', [])

                if frequencies and magnitudes:
                    for freq, mag in zip(frequencies, magnitudes):
                        combined_data.append({
                            'frequency': freq,
                            'magnitude': mag,
                            'phase': name
                        })

        return pd.DataFrame(combined_data)

    @staticmethod
    def calculate_health_score(features: Dict, anomalies: List[Dict]) -> float:
        """Расчет индекса здоровья оборудования"""
        if not features:
            return 50.0  # Нейтральное значение при отсутствии данных

        score = 100.0

        # Анализ RMS значений
        for phase in ['a', 'b', 'c']:
            rms_key = f'rms_{phase}'
            if rms_key in features:
                rms_val = features[rms_key]
                # Нормализация RMS (предполагаем нормальный диапазон 0-50А)
                if rms_val > 50:
                    score -= 10
                elif rms_val > 40:
                    score -= 5

        # Анализ Crest Factor
        for phase in ['a', 'b', 'c']:
            crest_key = f'crest_{phase}'
            if crest_key in features:
                crest_val = features[crest_key]
                # Нормальный Crest Factor 1.2-1.8
                if crest_val > 2.0 or crest_val < 1.0:
                    score -= 8

        # Анализ аномалий
        recent_anomalies = [
            a for a in anomalies
            if a.get('is_anomaly') and
            self._is_recent_event(a.get('created_at'))
        ]

        anomaly_penalty = len(recent_anomalies) * 15
        score = max(0, score - anomaly_penalty)

        return min(100.0, max(0.0, score))

    @staticmethod
    def _is_recent_event(event_time: Optional[str]) -> bool:
        """Проверка, что событие произошло недавно (последние 24 часа)"""
        if not event_time:
            return False

        try:
            event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
            return (datetime.now() - event_dt).total_seconds() < 86400  # 24 часа
        except:
            return False

class ValidationUtils:
    """Утилиты для валидации данных"""

    @staticmethod
    def validate_signal_data(signal_data: Dict) -> bool:
        """Валидация данных сигнала"""
        required_fields = ['sample_rate_hz', 'samples_count']

        if not all(field in signal_data for field in required_fields):
            return False

        # Проверка наличия хотя бы одной фазы
        phases = ['phase_a', 'phase_b', 'phase_c']
        has_phase_data = any(
            phase in signal_data and signal_data[phase]
            for phase in phases
        )

        return has_phase_data

    @staticmethod
    def validate_features(features: Dict) -> bool:
        """Валидация признаков"""
        if not features:
            return False

        # Проверка наличия базовых статистических признаков
        required_features = []
        phases = ['a', 'b', 'c']

        for phase in phases:
            required_features.extend([
                f'rms_{phase}',
                f'crest_{phase}',
                f'kurt_{phase}',
                f'skew_{phase}'
            ])

        # Достаточно наличия признаков хотя бы для одной фазы
        phase_features_count = sum(
            1 for feature in required_features[:4]
            if feature in features
        )

        return phase_features_count >= 2  # Минимум 2 признака для одной фазы

class FormatUtils:
    """Утилиты для форматирования данных"""

    @staticmethod
    def format_timestamp(timestamp: str) -> str:
        """Форматирование временной метки"""
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%d.%m.%Y %H:%M:%S')
        except:
            return timestamp

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Форматирование размера файла"""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Форматирование длительности"""
        if seconds < 60:
            return f"{seconds:.1f} сек"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} мин"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} ч"

    @staticmethod
    def format_number(value: float, decimals: int = 3) -> str:
        """Форматирование числовых значений"""
        if value is None:
            return "N/A"

        if abs(value) >= 1000:
            return f"{value:,.{decimals}f}"
        else:
            return f"{value:.{decimals}f}"

class SecurityUtils:
    """Утилиты безопасности для Dashboard"""

    @staticmethod
    def hash_sensitive_data(data: str) -> str:
        """Хеширование чувствительных данных"""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Очистка имени файла от опасных символов"""
        import re
        # Удаляем или заменяем опасные символы
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Ограничиваем длину
        return safe_filename[:100]

    @staticmethod
    def validate_user_permission(user_role: str, required_role: str) -> bool:
        """Проверка прав пользователя"""
        role_hierarchy = {
            'admin': 3,
            'engineer': 2,
            'operator': 1,
            'guest': 0
        }

        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        return user_level >= required_level

class ConfigManager:
    """Менеджер конфигурации Dashboard"""

    @staticmethod
    def load_dashboard_config() -> Dict:
        """Загрузка конфигурации Dashboard"""
        default_config = {
            'api_base_url': 'http://api:8000',
            'refresh_intervals': [10, 30, 60, 120, 300],
            'chart_themes': [
                'plotly', 'plotly_white', 'plotly_dark',
                'ggplot2', 'seaborn', 'simple_white'
            ],
            'max_chart_points': 10000,
            'export_formats': ['csv', 'json', 'pdf'],
            'session_timeout': 3600  # 1 час
        }

        return default_config

    @staticmethod
    def save_user_preferences(user_id: str, preferences: Dict):
        """Сохранение пользовательских настроек"""
        # В реальном приложении здесь была бы работа с базой данных
        preferences_key = f"user_preferences_{user_id}"
        st.session_state[preferences_key] = preferences

    @staticmethod
    def load_user_preferences(user_id: str) -> Dict:
        """Загрузка пользовательских настроек"""
        preferences_key = f"user_preferences_{user_id}"
        return st.session_state.get(preferences_key, {})
