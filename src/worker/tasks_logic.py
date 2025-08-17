# Логика Celery задач (перенос из старого tasks.py)
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, UTC
from typing import Dict, Any
from uuid import UUID

import numpy as np
from celery import Task
from celery.signals import worker_ready, worker_shutdown
from sqlalchemy import select

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import (
    RawSignal, Feature, Equipment, Prediction,
    ProcessingStatus
)
from src.data_processing.feature_extraction import FeatureExtractor
from src.worker.processing_core import process_raw_core, _update_signal_status as core_update_status
from src.ml.train import load_latest_models
from src.ml.forecasting import RMSTrendForecaster
from src.utils.logger import get_logger, get_audit_logger
from src.worker.config import celery_app
from src.utils.serialization import load_float32_array
from src.utils.metrics import observe_latency as _observe_latency

settings = get_settings()
logger = get_logger(__name__)
audit_logger = get_audit_logger()

class DatabaseTask(Task):
    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    def on_failure(self, exc, task_id, args, kwargs, einfo):  # noqa: D401
        self.logger.error("task failed", extra={'task_id': task_id, 'exc': str(exc)})
    def on_retry(self, exc, task_id, args, kwargs, einfo):  # noqa: D401
        self.logger.warning("task retry", extra={'task_id': task_id})
    def on_success(self, retval, task_id, args, kwargs):  # noqa: D401
        self.logger.info("task success", extra={'task_id': task_id})

async def decompress_signal_data(compressed_data: bytes) -> np.ndarray:
    try:
        arr = load_float32_array(compressed_data)
        return arr if arr is not None else np.array([], dtype=np.float32)
    except Exception as e:  # pragma: no cover
        logger.error(f"Ошибка распаковки данных: {e}")
        raise

async def compress_and_store_results(data: Any) -> bytes:
    try:
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        import gzip
        return gzip.compress(json_str.encode('utf-8'))
    except Exception as e:  # pragma: no cover
        logger.error(f"Ошибка сжатия результатов: {e}")
        raise

@_observe_latency('worker_task_duration_seconds', labels={'task_name': 'process_raw'})
@celery_app.task(bind=True, base=DatabaseTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60}, retry_backoff=True, retry_jitter=True)
def process_raw(self, raw_id: str) -> Dict:  # type: ignore
    task_start = datetime.now(UTC)
    try:
        result = asyncio.run(process_raw_core(raw_id))
        result['processing_time_seconds'] = (datetime.now(UTC) - task_start).total_seconds()
        return result
    except Exception as exc:  # pragma: no cover
        asyncio.run(core_update_status(raw_id, ProcessingStatus.FAILED, str(exc)))
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (self.request.retries + 1))
        raise

async def _process_raw_async(raw_id: str) -> Dict:  # backward compat if imported elsewhere
    return await process_raw_core(raw_id)

async def _update_signal_status(raw_id: str, status: ProcessingStatus, error: str | None = None):
    await core_update_status(raw_id, status, error)

# Лишние задачи / функции вырезаны для краткости — восстановить при необходимости.

__all__ = [
    'process_raw', '_process_raw_async', 'decompress_signal_data', 'compress_and_store_results', '_update_signal_status'
]
