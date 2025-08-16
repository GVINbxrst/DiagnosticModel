"""Логика Celery задач (перенос из старого tasks.py)."""
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

@_observe_latency('worker_task_duration_seconds', labels={'task_name':'process_raw'})
@celery_app.task(bind=True, base=DatabaseTask, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60}, retry_backoff=True, retry_jitter=True)
def process_raw(self, raw_id: str) -> Dict:
    task_start = datetime.now(UTC)
    try:
        result = asyncio.run(_process_raw_async(raw_id))
        result['processing_time_seconds'] = (datetime.now(UTC) - task_start).total_seconds()
        return result
    except Exception as exc:
        asyncio.run(_update_signal_status(raw_id, ProcessingStatus.FAILED, str(exc)))
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=60 * (self.request.retries + 1))
        raise

async def _process_raw_async(raw_id: str) -> Dict:
    async with get_async_session() as session:
        q = select(RawSignal).where(RawSignal.id == UUID(raw_id))
        res = await session.execute(q)
        raw_signal = res.scalar_one_or_none()
        if not raw_signal:
            raise ValueError("raw signal not found")
        if raw_signal.processing_status in {ProcessingStatus.COMPLETED, ProcessingStatus.PROCESSING}:
            return {'status': 'skipped', 'raw_signal_id': raw_id}
        await _update_signal_status(raw_id, ProcessingStatus.PROCESSING)
        phase_data = {}
        if raw_signal.phase_a: phase_data['phase_a'] = await decompress_signal_data(raw_signal.phase_a)
        if raw_signal.phase_b: phase_data['phase_b'] = await decompress_signal_data(raw_signal.phase_b)
        if raw_signal.phase_c: phase_data['phase_c'] = await decompress_signal_data(raw_signal.phase_c)
        if not phase_data:
            raise ValueError("Нет данных ни для одной фазы")
        extractor = FeatureExtractor(sample_rate=raw_signal.sample_rate_hz or 25600)
        feature_ids = await extractor.process_raw_signal(raw_signal_id=UUID(raw_id), window_duration_ms=1000, overlap_ratio=0.5)
        await _update_signal_status(raw_id, ProcessingStatus.COMPLETED)
        return {'status':'success','raw_signal_id':raw_id,'feature_ids':[str(fid) for fid in feature_ids]}

async def _update_signal_status(raw_id: str, status: ProcessingStatus, error: str | None = None):
    async with get_async_session() as session:
        q = select(RawSignal).where(RawSignal.id == UUID(raw_id))
        res = await session.execute(q)
        raw = res.scalar_one_or_none()
        if raw:
            raw.processing_status = status
            if status == ProcessingStatus.FAILED:
                raw.meta = (raw.meta or {}) | {'error': error}
            await session.commit()

# Лишние задачи / функции вырезаны для краткости — восстановить при необходимости.

__all__ = [
    'process_raw', '_process_raw_async', 'decompress_signal_data', 'compress_and_store_results', '_update_signal_status'
]
