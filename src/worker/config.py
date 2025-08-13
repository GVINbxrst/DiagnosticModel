"""
Конфигурация и инициализация Celery worker для диагностики двигателей

Этот модуль настраивает Celery worker с оптимальными параметрами
для обработки данных диагностики асинхронных двигателей.
"""

import os
import signal
import sys
from typing import Dict, Any

from celery import Celery
from celery.signals import setup_logging
from kombu import Queue

from src.config.settings import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)


def create_celery_app() -> Celery:
    """
    Создание и настройка Celery приложения

    Returns:
        Настроенное Celery приложение
    """

    # Создаем Celery приложение
    celery_app = Celery('diagmod_worker')

    # Конфигурация брокера и backend
    celery_app.conf.update(
        # Соединения
        broker_url=settings.REDIS_URL,
        result_backend=settings.REDIS_URL,

        # Сериализация
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',

        # Временные зоны
        timezone='UTC',
        enable_utc=True,

        # Отслеживание задач
        task_track_started=True,
        task_send_sent_event=True,
        worker_send_task_events=True,

        # Лимиты времени
        task_time_limit=3600,      # 1 час жесткий лимит
        task_soft_time_limit=3300, # 55 минут мягкий лимит

        # Настройки worker
        worker_prefetch_multiplier=1,  # Берем по одной задаче
        task_acks_late=True,           # Подтверждаем после выполнения
        worker_disable_rate_limits=False,

        # Сжатие
        task_compression='gzip',
        result_compression='gzip',

        # Маршрутизация задач по очередям
        task_routes={
            # Обработка данных - высокий приоритет, долгие задачи
            'src.worker.tasks.process_raw': {
                'queue': 'processing',
                'routing_key': 'processing'
            },

            # ML задачи - средний приоритет, требуют больше памяти
            'src.worker.tasks.detect_anomalies': {
                'queue': 'ml',
                'routing_key': 'ml'
            },
            'src.worker.tasks.forecast_trend': {
                'queue': 'ml',
                'routing_key': 'ml'
            },
            'src.worker.tasks.retrain_models': {
                'queue': 'ml',
                'routing_key': 'ml'
            },

            # Пакетные задачи - низкий приоритет
            'src.worker.specialized_tasks.batch_process_directory': {
                'queue': 'batch',
                'routing_key': 'batch'
            },
            'src.worker.specialized_tasks.process_equipment_workflow': {
                'queue': 'batch',
                'routing_key': 'batch'
            },

            # Служебные задачи
            'src.worker.tasks.cleanup_old_data': {
                'queue': 'maintenance',
                'routing_key': 'maintenance'
            },
            'src.worker.specialized_tasks.health_check_system': {
                'queue': 'monitoring',
                'routing_key': 'monitoring'
            },
            'src.worker.specialized_tasks.daily_equipment_report': {
                'queue': 'monitoring',
                'routing_key': 'monitoring'
            }
        },

        # Определение очередей с приоритетами
        task_queues=(
            # Высокий приоритет - обработка сырых данных
            Queue('processing', routing_key='processing', queue_arguments={
                'x-max-priority': 10,
                'x-message-ttl': 3600000  # 1 час TTL
            }),

            # Средний приоритет - ML задачи
            Queue('ml', routing_key='ml', queue_arguments={
                'x-max-priority': 7,
                'x-message-ttl': 7200000  # 2 часа TTL
            }),

            # Низкий приоритет - пакетные операции
            Queue('batch', routing_key='batch', queue_arguments={
                'x-max-priority': 3,
                'x-message-ttl': 86400000  # 24 часа TTL
            }),

            # Служебные задачи
            Queue('maintenance', routing_key='maintenance', queue_arguments={
                'x-max-priority': 2,
                'x-message-ttl': 86400000
            }),

            # Мониторинг
            Queue('monitoring', routing_key='monitoring', queue_arguments={
                'x-max-priority': 5,
                'x-message-ttl': 3600000
            })
        ),

        # Настройки retry
        task_retry_delay=60,       # Базовая задержка между повторами
        task_max_retries=3,        # Максимум повторов по умолчанию

        # Результаты
        result_expires=3600,       # Результаты хранятся 1 час

        # Мониторинг
        worker_hijack_root_logger=False,  # Не перехватываем корневой логгер

        # Beat scheduler для периодических задач
        beat_schedule={
            # Очистка старых данных каждый час
            'cleanup-old-data': {
                'task': 'src.worker.tasks.cleanup_old_data',
                'schedule': 3600.0,
                'options': {'queue': 'maintenance'}
            },

            # Переобучение моделей каждый день в 2:00 UTC
            'retrain-models': {
                'task': 'src.worker.tasks.retrain_models',
                'schedule': 86400.0,
                'options': {'queue': 'ml'}
            },

            # Проверка здоровья системы каждые 15 минут
            'health-check': {
                'task': 'src.worker.specialized_tasks.health_check_system',
                'schedule': 900.0,
                'options': {'queue': 'monitoring'}
            },

            # Ежедневный отчет в 8:00 UTC
            'daily-report': {
                'task': 'src.worker.specialized_tasks.daily_equipment_report',
                'schedule': 86400.0,
                'options': {'queue': 'monitoring'}
            }
        }
    )

    # Автоматическое обнаружение задач
    celery_app.autodiscover_tasks([
        'src.worker.tasks',
    'src.worker.specialized_tasks'
    ])

    return celery_app


def setup_worker_signals(celery_app: Celery):
    """Настройка обработчиков сигналов для worker"""

    def shutdown_handler(signum, frame):
        """Обработчик graceful shutdown"""
        logger.info(f"Получен сигнал {signum}, начинаем graceful shutdown")

        # Останавливаем прием новых задач
        celery_app.control.cancel_consumer('processing')
        celery_app.control.cancel_consumer('ml')
        celery_app.control.cancel_consumer('batch')

        logger.info("Worker завершает работу...")
        sys.exit(0)

    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, shutdown_handler)


@setup_logging.connect
def setup_celery_logging(**kwargs):
    """Настройка логирования для Celery"""
    # Локальная функция оставлена для совместимости; логгер уже настроен в src.utils.logger
    return None


def get_worker_info() -> Dict[str, Any]:
    """
    Получение информации о конфигурации worker

    Returns:
        Информация о настройках worker
    """
    return {
        'broker_url': settings.REDIS_URL,
        'queues': ['processing', 'ml', 'batch', 'maintenance', 'monitoring'],
        'task_time_limit': 3600,
        'soft_time_limit': 3300,
        'prefetch_multiplier': 1,
        'compression': 'gzip',
        'timezone': 'UTC'
    }


# Создаем экземпляр Celery приложения
celery_app = create_celery_app()

# Настраиваем сигналы
setup_worker_signals(celery_app)

# Экспорт для использования
__all__ = ['celery_app', 'get_worker_info']
