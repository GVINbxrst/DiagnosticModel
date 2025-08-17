# Минимальная конфигурация Celery (имя diagmod, брокер/бэкенд из settings)

from celery import Celery

from src.config.settings import get_settings
import os

settings = get_settings()

celery_app = Celery(
    'diagmod',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'src.worker.tasks',
        'src.worker.specialized_tasks',
    ]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    broker_transport_options={
        'visibility_timeout': 7200,
    },
    result_expires=86400,
    task_default_retry_delay=60,
    task_routes={
        'src.worker.tasks.process_raw': {'queue': 'processing'},
        'src.worker.tasks.cleanup_old_data': {'queue': 'maintenance'},
        'src.worker.tasks.retrain_models': {'queue': 'ml'},
    },
)

celery_app.autodiscover_tasks([
    'src.worker',
])

# Локальный режим без брокера (eager) при отладке / E2E скриптах:
if os.getenv('CELERY_TASK_ALWAYS_EAGER', '0').lower() in {'1', 'true', 'yes'}:
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    # Явный лог в stdout (через print чтобы увидеть даже до инициализации логгера)
    print('[Celery] task_always_eager=TRUE (локальный режим без брокера)')  # pragma: no cover


def get_worker_info() -> dict:
    # Возвращает краткую информацию о воркере (для обратной совместимости)
    return {
        "app": celery_app.main or "diagmod",
        "broker": celery_app.connection().as_uri() if celery_app.connection() else settings.CELERY_BROKER_URL,
        "backend": settings.CELERY_RESULT_BACKEND,
        "timezone": celery_app.conf.timezone,
        "utc": celery_app.conf.enable_utc,
    }


__all__ = ["celery_app", "get_worker_info"]
