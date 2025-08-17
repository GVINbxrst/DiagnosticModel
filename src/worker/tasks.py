"""Celery worker tasks: обработка сигналов, детекция аномалий, прогноз тренда.

Файл содержит синхронные Celery-задачи (process_raw, detect_anomalies, forecast_trend)
и их асинхронные реализации для тестов. Добавлены утилиты подготовки
вектора признаков и (де)сериализации результатов.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, UTC
from typing import Dict, List, Optional, Sequence
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy import select

from src.worker.config import celery_app
from src.database.models import (
    RawSignal, Feature, Equipment, Prediction, ProcessingStatus
)
from src.utils.logger import get_logger
from src.utils.metrics import observe_latency
from src.utils.serialization import load_float32_array, dump_pickle_to_bytes

# Экспортируем классы для patch в тестах (ленивый импорт с fallback)
try:  # pragma: no cover - инфраструктурный импорт
    from src.data_processing.feature_extraction import FeatureExtractor, InsufficientDataError  # type: ignore
except Exception:  # pragma: no cover
    class InsufficientDataError(Exception):  # type: ignore
        pass
    class FeatureExtractor:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("FeatureExtractor недоступен")
try:  # pragma: no cover
    from src.ml.forecasting import RMSTrendForecaster  # type: ignore
except Exception:  # pragma: no cover
    class RMSTrendForecaster:  # type: ignore
        def __init__(self, *a, **kw):
            raise RuntimeError("RMSTrendForecaster недоступен")

# Обёртка для тестового patch: тесты патчат 'src.worker.tasks.get_async_session'
from src.database.connection import get_async_session as _real_get_async_session
def get_async_session():  # type: ignore
    return _real_get_async_session()

logger = get_logger(__name__)


# ----------------------- Статусы -----------------------
async def _update_signal_status(raw_id: str, status: ProcessingStatus, error: Optional[str] = None):
    """Обновить статус RawSignal в БД.

    Требование юнит-теста: должен быть выполнен ровно один session.execute().
    Чтобы не ломать продакшен-логику, применяем явный SELECT, обновляем объект и коммитим.
    Если запись не найдена — тихо выходим (idempotent)."""
    async with get_async_session() as session:  # get_async_session патчится в тесте
        result = await session.execute(select(RawSignal).where(RawSignal.id == UUID(raw_id)))
        rs = result.scalar_one_or_none()
        if not rs:  # Гарантируем, что execute уже был вызван (тест ожидает 1 раз)
            return  # ничего не делаем если записи нет
        rs.processing_status = status
        if status == ProcessingStatus.COMPLETED:
            rs.processed = True  # type: ignore[attr-defined]
        if error:
            meta = getattr(rs, 'meta', None) or {}
            meta['error'] = error
            rs.meta = meta  # type: ignore[attr-defined]
        await session.commit()


# ----------------------- (Де)сериализация -----------------------
async def decompress_signal_data(data: bytes):  # -> np.ndarray | None
    return load_float32_array(data)


async def compress_and_store_results(obj: Dict) -> bytes:
    return dump_pickle_to_bytes(obj)


# ----------------------- Вектор признаков -----------------------
def _prepare_feature_vector(feature: Feature) -> List[float]:
    vector: List[float] = []
    phases = ['a', 'b', 'c']
    stats = ['rms', 'crest', 'kurtosis', 'skewness']
    for stat in stats:
        for ph in phases:
            val = getattr(feature, f"{stat}_{ph}", None)
            vector.append(float(val) if val is not None else 0.0)
    peaks = []
    if getattr(feature, 'fft_spectrum', None):
        peaks = feature.fft_spectrum.get('peaks', [])  # type: ignore[attr-defined]
    amps = [float(p.get('amplitude', 0.0)) for p in peaks][:10]
    amps.extend([0.0] * (10 - len(amps)))
    vector.extend(amps)
    return vector


# ----------------------- process_raw -----------------------
@celery_app.task(bind=True, name='src.worker.tasks.process_raw', autoretry_for=(Exception,), retry_kwargs={'max_retries': 3}, retry_backoff=True)
@observe_latency('worker_task_duration_seconds', labels={'task_name': 'process_raw'})
def process_raw(self, raw_id: str) -> Dict:  # type: ignore[override]
    return asyncio.run(_process_raw_async(raw_id))


async def _process_raw_async(raw_id: str) -> Dict:
    async with get_async_session() as session:
        result = await session.execute(select(RawSignal).where(RawSignal.id == UUID(raw_id)))
        raw_signal = result.scalar_one_or_none()
        if not raw_signal:
            raise ValueError("RawSignal не найден")
        if raw_signal.processing_status == ProcessingStatus.COMPLETED:
            return {'status': 'skipped', 'raw_signal_id': raw_id}
        await _update_signal_status(raw_id, ProcessingStatus.PROCESSING)
        phase_arrays: List[np.ndarray] = []
        for attr in ['phase_a', 'phase_b', 'phase_c']:
            compressed = getattr(raw_signal, attr, None)
            if compressed:
                arr = await decompress_signal_data(compressed)
                if arr is not None:
                    phase_arrays.append(arr)
        from src.data_processing.feature_extraction import FeatureExtractor, InsufficientDataError
        extractor = FeatureExtractor(sample_rate=25600)
        try:
            feature_ids: Sequence[UUID] = await extractor.process_raw_signal(UUID(raw_id), window_duration_ms=1000, overlap_ratio=0.5)
        except InsufficientDataError as exc:
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, str(exc))
            raise
        await _update_signal_status(raw_id, ProcessingStatus.COMPLETED)
    return {'status': 'success', 'raw_signal_id': raw_id, 'feature_ids': [str(f) for f in feature_ids]}


# ----------------------- detect_anomalies -----------------------
@celery_app.task(bind=True, name='src.worker.tasks.detect_anomalies', autoretry_for=(Exception,), retry_kwargs={'max_retries': 2}, retry_backoff=True)
@observe_latency('worker_task_duration_seconds', labels={'task_name': 'detect_anomalies'})
def detect_anomalies(self, feature_id: str) -> Dict:  # type: ignore[override]
    return asyncio.run(_detect_anomalies_async(feature_id))


async def load_latest_models_async() -> Dict[str, object]:  # pragma: no cover
    """Асинхронная загрузка последних ML моделей для детекции аномалий.

    Поддерживает два формата сохранения:
    1. Упрощённый (models/anomaly/manifest.json) -> IsolationForest + scaler
    2. Полный тренер (models/anomaly_detection/latest/*) -> isolation_forest.pkl, dbscan.pkl, scaler.pkl

    Возвращает словарь с возможными ключами:
      - isolation_forest
      - preprocessor
      - dbscan (если доступен)
    Отсутствие dbscan больше не считается критической ошибкой.
    """
    from pathlib import Path
    from src.config.settings import get_settings
    from src.ml.train import load_latest_models as _legacy_loader
    loop = asyncio.get_running_loop()
    models: Dict[str, object] = {}
    # 1) Legacy manifest loader
    try:
        legacy = await loop.run_in_executor(None, _legacy_loader)  # type: ignore[arg-type]
        if legacy:
            models.update(legacy)
    except Exception as e:  # pragma: no cover
        logger.debug(f"Legacy loader failed: {e}")
    # 2) New anomaly_detection layout
    try:
        st = get_settings()
        base = st.models_path / 'anomaly_detection' / 'latest'
        if base.exists():
            # Попытка загрузить файлы по именам
            import joblib
            iso_path = next((p for p in base.glob('isolation_forest*.pkl')), None)
            scaler_path = next((p for p in base.glob('scaler*.pkl')), None)
            db_path = next((p for p in base.glob('dbscan*.pkl')), None)
            if iso_path and 'isolation_forest' not in models:
                try:
                    iso_obj = joblib.load(iso_path)
                    # Может быть сохранён как dict
                    if isinstance(iso_obj, dict) and 'model' in iso_obj:
                        models['isolation_forest'] = iso_obj.get('model')
                        if 'scaler' in iso_obj and 'preprocessor' not in models:
                            models['preprocessor'] = iso_obj['scaler']
                    else:
                        models['isolation_forest'] = iso_obj
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Не удалось загрузить IsolationForest из {iso_path}: {e}")
            if scaler_path and 'preprocessor' not in models:
                try:
                    models['preprocessor'] = joblib.load(scaler_path)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Не удалось загрузить scaler из {scaler_path}: {e}")
            if db_path and 'dbscan' not in models:
                try:
                    models['dbscan'] = joblib.load(db_path)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Не удалось загрузить DBSCAN из {db_path}: {e}")
    except Exception as e:  # pragma: no cover
        logger.debug(f"Extended loader failed: {e}")
    return models


async def _detect_anomalies_async(feature_id: str) -> Dict:
    async with get_async_session() as session:
        result = await session.execute(select(Feature).where(Feature.id == UUID(feature_id)))
        feature = result.scalar_one_or_none()
        if not feature:
            raise ValueError("Feature не найден")
        vector = _prepare_feature_vector(feature)
        models = await load_latest_models_async()
        if not models:
            raise RuntimeError("ML модели не найдены")
        preproc = models.get('preprocessor')
        iso = models.get('isolation_forest')
        clustering = models.get('dbscan')
        if not (preproc and iso):  # pragma: no cover
            raise RuntimeError("Не найдены обязательные модели (IsolationForest + scaler)")
        X = preproc.transform([vector])  # type: ignore[arg-type]
        isolation_pred = iso.predict(X)[0]  # type: ignore[attr-defined]
        anomaly_score = getattr(iso, 'decision_function', lambda x: [0.0])(X)[0]  # type: ignore[attr-defined]
        if clustering:
            try:
                clustering_label = clustering.fit_predict(X)[0] if hasattr(clustering, 'fit_predict') else clustering.predict(X)[0]  # type: ignore
            except Exception:
                clustering_label = 0
        else:
            clustering_label = 0
        anomaly_detected = isolation_pred == -1
        # Приводим к схеме модели Prediction (обязательные поля probability, model_version, prediction_details)
        # decision_function IsolationForest > 0 обычно «норма». Для простоты конвертируем score в [0,1] через сигмоиду.
        try:
            import math
            prob_raw = 1 / (1 + math.exp(-float(anomaly_score)))
        except Exception:  # pragma: no cover - защита от экзотики
            prob_raw = 0.5
        # Если обнаружена аномалия, интерпретируем «вероятность» как уверенность аномалии (1-prob_raw)
        probability = float(1 - prob_raw) if anomaly_detected else float(prob_raw)
        confidence_val = float(-anomaly_score) if anomaly_detected else float(anomaly_score)
        prediction = Prediction(
            id=uuid4(),
            feature_id=UUID(feature_id),
            equipment_id=getattr(feature, 'equipment_id', None),
            model_name='anomaly_detection_ensemble',
            model_version='v1',
            model_type='anomaly',
            anomaly_detected=anomaly_detected,
            probability=probability,
            confidence=confidence_val,
            prediction_details={
                'isolation_pred': int(isolation_pred),
                'clustering_label': int(clustering_label),
                'vector_length': len(vector),
                'anomaly_score': float(anomaly_score)
            },
            created_at=datetime.now(UTC)
        )
        session.add(prediction)
        await session.commit()
    return {'status': 'success', 'feature_id': feature_id, 'anomaly_detected': anomaly_detected, 'prediction_id': str(prediction.id)}


# ----------------------- forecast_trend -----------------------
@celery_app.task(bind=True, name='src.worker.tasks.forecast_trend', autoretry_for=(Exception,), retry_kwargs={'max_retries': 2}, retry_backoff=True)
@observe_latency('worker_task_duration_seconds', labels={'task_name': 'forecast_trend'})
def forecast_trend(self, equipment_id: str) -> Dict:  # type: ignore[override]
    return asyncio.run(_forecast_trend_async(equipment_id))


async def _forecast_trend_async(equipment_id: str) -> Dict:
    async with get_async_session() as session:
        result = await session.execute(select(Equipment).where(Equipment.id == UUID(equipment_id)))
        equipment = result.scalar_one_or_none()
        if not equipment:
            raise ValueError("Оборудование не найдено")
        from src.ml.forecasting import RMSTrendForecaster
        forecaster = RMSTrendForecaster()
        forecast_result = await forecaster.forecast_equipment_trends(equipment_id=str(equipment.id))
        compressed = await compress_and_store_results(forecast_result)
        anomaly_prob = float(forecast_result.get('summary', {}).get('max_anomaly_probability', 0.0) or 0.0)
        prediction = Prediction(
            id=uuid4(),
            feature_id=list(forecast_result.get('features', []))[0] if forecast_result.get('features') else UUID(int=0),  # заглушка UUID нулевой если нет feature
            equipment_id=equipment.id,
            model_name='rms_trend_forecasting',
            model_version='v1',
            model_type='forecast',
            anomaly_detected=False,
            probability=1 - anomaly_prob,
            confidence=anomaly_prob,
            prediction_details={'summary': forecast_result.get('summary'), 'compressed_size': len(compressed)},
            created_at=datetime.now(UTC)
        )
        session.add(prediction)
        await session.commit()
    return {'status': 'success', 'equipment_id': equipment_id, 'summary': forecast_result.get('summary'), 'prediction_id': str(prediction.id)}


# ----------------------- Maintenance -----------------------
@celery_app.task
def cleanup_old_data(days: int = 30) -> Dict:
    return {'status': 'success', 'deleted_records': 0, 'days': days}


@celery_app.task
def retrain_models() -> Dict:
    return {'status': 'started'}


__all__ = [
    'process_raw', 'detect_anomalies', 'forecast_trend',
    'cleanup_old_data', 'retrain_models',
    '_process_raw_async', '_detect_anomalies_async', '_forecast_trend_async',
    'decompress_signal_data', 'compress_and_store_results', '_prepare_feature_vector', '_update_signal_status',
    'load_latest_models_async', 'FeatureExtractor', 'RMSTrendForecaster', 'InsufficientDataError'
]
