"""
Celery Worker модуль для фоновой обработки данных диагностики двигателей

Этот модуль содержит:
- Конфигурацию Celery worker
- Основные задачи обработки данных
- Специализированные задачи мониторинга
"""

from src.worker.config import celery_app, get_worker_info
from src.worker.tasks import process_raw, detect_anomalies, forecast_trend, cleanup_old_data, retrain_models

__all__ = [
    'celery_app',
    'get_worker_info',
    'process_raw',
    'detect_anomalies',
    'forecast_trend',
    'cleanup_old_data',
    'retrain_models'
]
