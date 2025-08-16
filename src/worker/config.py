# Минимальная конфигурация Celery (имя diagmod, брокер/бэкенд из settings)

from celery import Celery

from src.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    'diagmod',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
)

celery_app.autodiscover_tasks([
    'src.worker.tasks'
])


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
