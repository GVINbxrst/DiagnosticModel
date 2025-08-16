# Система логирования: структурированное JSON логирование

import json
import logging
import logging.config
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

import structlog
try:  # Prefer new location to avoid deprecation
    from pythonjsonlogger import json as jsonlogger  # type: ignore
except Exception:  # fallback to old path
    from pythonjsonlogger import jsonlogger  # type: ignore

from src.config.settings import get_settings


class CustomJSONFormatter(jsonlogger.JsonFormatter):
    # Кастомный JSON форматтер с доп. полями

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)

        # Добавляем временную метку в ISO формате
        if not log_record.get('timestamp'):
            # Используем timezone-aware UTC вместо устаревшего utcnow
            log_record['timestamp'] = datetime.now(UTC).isoformat().replace('+00:00', 'Z')

        # Добавляем информацию о модуле
        if record.name:
            log_record['module'] = record.name

        # Добавляем уровень логирования
        if record.levelname:
            log_record['level'] = record.levelname

        # Добавляем информацию о приложении
        log_record['app'] = 'diagmod'

        # Добавляем информацию о процессе и потоке
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread

        # Добавляем трассировку для ошибок
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }


class AuditLogger:
    # Логгер аудита действий пользователей

    def __init__(self):
        self.logger = logging.getLogger('diagmod.audit')

    def log_user_action(self, user_id: str, username: str, action: str,
                       resource: str, result: str, ip_address: str = None,
                       user_agent: str = None, additional_data: Dict = None):
        # Лог действия пользователя
        audit_data = {
            'event_type': 'user_action',
            'user_id': user_id,
            'username': username,
            'action': action,
            'resource': resource,
            'result': result,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'additional_data': additional_data or {}
        }
        self.logger.info("User action logged", extra=audit_data)

    def log_api_request(self, user_id: str, endpoint: str, method: str,
                       status_code: int, response_time: float, ip_address: str):
        # Лог API запроса
        api_data = {
            'event_type': 'api_request',
            'user_id': user_id,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': response_time * 1000,
            'ip_address': ip_address
        }
        self.logger.info("API request", extra=api_data)

    def log_anomaly_detection(self, equipment_id: int, model_name: str,
                             is_anomaly: bool, confidence: float, features: Dict):
        # Лог обнаружения аномалий
        anomaly_data = {
            'event_type': 'anomaly_detection',
            'equipment_id': equipment_id,
            'model_name': model_name,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'features': features
        }
        self.logger.info("Anomaly detection completed", extra=anomaly_data)

    def log_forecast_generation(self, equipment_id: int, model_name: str,
                               forecast_horizon: int, execution_time: float):
        # Лог генерации прогноза
        forecast_data = {
            'event_type': 'forecast_generation',
            'equipment_id': equipment_id,
            'model_name': model_name,
            'forecast_horizon': forecast_horizon,
            'execution_time_seconds': execution_time
        }
        self.logger.info("Forecast generated", extra=forecast_data)


def setup_logging():
    # Настроить систему логирования
    settings = get_settings()

    # Создаем директорию для логов
    if settings.log_file_path:
        settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Конфигурация логирования
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': CustomJSONFormatter,
                'format': '%(timestamp)s %(level)s %(module)s %(message)s'
            },
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.log_level,
                'formatter': 'json' if settings.log_format == 'json' else 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            '': {  # root logger
                'level': settings.log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'diagmod': {
                'level': settings.log_level,
                'handlers': ['console'],
                'propagate': False
            },
            'diagmod.audit': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            },
            'celery': {
                'level': 'INFO',
                'handlers': ['console'],
                'propagate': False
            }
        }
    }

    # Добавляем файловый обработчик, если указан путь к файлу
    if settings.log_file_path:
        log_config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': settings.log_level,
            'formatter': 'json',
            'filename': str(settings.log_file_path),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        }

        # Добавляем файловый обработчик ко всем логгерам
        for logger_name in log_config['loggers']:
            log_config['loggers'][logger_name]['handlers'].append('file')

    # Применяем конфигурацию
    logging.config.dictConfig(log_config)

    # Настраиваем structlog для структурированного логирования
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.environment == "development" else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> logging.Logger:
    # Получить настроенный логгер
    if not name:
        name = 'diagmod'
    return logging.getLogger(name)


def get_audit_logger() -> AuditLogger:
    # Получить логгер аудита
    return AuditLogger()


# Глобальные экземпляры
logger = get_logger()
audit_logger = get_audit_logger()


if __name__ == "__main__":
    # Тест системы логирования
    logger = get_logger(__name__)

    logger.info("Тестирование стандартного логгера")
    logger.warning("Предупреждение с дополнительными данными", extra={"test": "value"})
    logger.error("Ошибка для тестирования")

    # Пример structured через structlog
    s_logger = structlog.get_logger(__name__)
    s_logger.info("Тестирование структурированного логгера", key="value", number=42)
    s_logger.warning("Структурированное предупреждение", module="test", action="log_test")

    print("Тест логирования завершен")
