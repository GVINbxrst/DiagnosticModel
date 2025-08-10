"""
DiagMod - Система токовой диагностики асинхронных двигателей

Основной пакет содержит модули для:
- Обработки CSV данных с токовыми сигналами
- Извлечения признаков (FFT, RMS, статистические моменты)
- ML моделей обнаружения аномалий и прогнозирования
- API для интеграции с внешними системами
- Dashboard для визуализации и мониторинга
"""

__version__ = "1.0.0"
__author__ = "DiagMod Team"
__email__ = "team@diagmod.com"

# Импорты основных компонентов
from src.config.settings import get_settings
from src.utils.logger import get_logger

# Получение настроек и логгера для инициализации
settings = get_settings()
logger = get_logger(__name__)

logger.info(f"DiagMod v{__version__} initialized")
