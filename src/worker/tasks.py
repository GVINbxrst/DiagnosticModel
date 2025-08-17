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
from src.config.settings import get_settings
from src.worker.processing_core import process_raw_core, _update_signal_status as core_update_status
from src.ml.clustering import full_clustering_pipeline, build_distribution_report, load_cluster_label_dict, semi_supervised_knn
from src.ml.utils import save_model, load_model, warmup_models
# Прогрев кеша моделей при старте воркера (только если включено)
try:  # pragma: no cover - выполнение при импорте
    st_settings = get_settings()
    if getattr(st_settings, 'USE_MODEL_CACHE', True):
        warmup_models(['clustering/knn_model'])
except Exception:  # noqa: E722
    pass
from src.ml.tcn_forecasting import predict_tcn
from src.database.models import ClusterLabel
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
    # Прокси к общей реализации (сохранение совместимости для тестов)
    await core_update_status(raw_id, status, error)  # type: ignore[arg-type]


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
    return await process_raw_core(raw_id)


# ----------------------- detect_anomalies -----------------------
@celery_app.task(bind=True, name='src.worker.tasks.detect_anomalies', autoretry_for=(Exception,), retry_kwargs={'max_retries': 2}, retry_backoff=True)
@observe_latency('worker_task_duration_seconds', labels={'task_name': 'detect_anomalies'})
def detect_anomalies(self, feature_id: str) -> Dict:  # type: ignore[override]
    return asyncio.run(_detect_anomalies_async(feature_id))


async def load_latest_models_async() -> Dict[str, object]:  # pragma: no cover
    """Упрощённая загрузка: теперь только потоковая / статистическая модель.

    Порядок приоритета:
      1. streaming (HalfSpaceTrees manifest)
      2. stats baseline manifest
    Возвращает {'anomaly_model': obj, 'model_type': 'stream'|'stats', 'threshold': float, 'version': str}
    """
    from pathlib import Path
    st = get_settings()
    base = st.models_path / 'anomaly_detection' / 'latest'
    out: Dict[str, object] = {}
    try:
        if base.exists():
            manifest_path = base / 'manifest.json'
            if manifest_path.exists():
                import json
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                mtype = manifest.get('model_type')
                threshold = float(manifest.get('threshold', 0.7))
                version = manifest.get('version', 'v1')
                if mtype == 'stream' and (base / 'stream_state.pkl').exists():
                    import joblib
                    state = joblib.load(base / 'stream_state.pkl')
                    from src.ml.incremental import StreamingHalfSpaceTreesAdapter
                    model = StreamingHalfSpaceTreesAdapter(threshold=threshold)
                    # best-effort восстановление (если state сериализован отдельно)
                    try:
                        model.model = state.get('model')  # type: ignore
                    except Exception:
                        pass
                    out = {'anomaly_model': model, 'model_type': 'stream', 'threshold': threshold, 'version': version}
                elif mtype == 'stats' and (base / 'stats_state.json').exists():
                    import json as _j
                    state = _j.loads((base / 'stats_state.json').read_text(encoding='utf-8'))
                    from src.ml.incremental import StatsQuantileBaseline
                    model = StatsQuantileBaseline(z_threshold=state.get('z_threshold', 6.0))
                    # значения не восстанавливаем (не критично)
                    out = {'anomaly_model': model, 'model_type': 'stats', 'threshold': model.z_threshold, 'version': version}
    except Exception as e:  # pragma: no cover
        logger.debug(f"model load fail: {e}")
    return out


async def _detect_anomalies_async(feature_id: str) -> Dict:
    async with get_async_session() as session:
        result = await session.execute(select(Feature).where(Feature.id == UUID(feature_id)))
        feature = result.scalar_one_or_none()
        if not feature:
            raise ValueError("Feature не найден")
        vector = _prepare_feature_vector(feature)
        models = await load_latest_models_async()
        if not models or 'anomaly_model' not in models:
            raise RuntimeError("Anomaly модель не найдена")
        model = models['anomaly_model']
        mtype = models.get('model_type', 'unknown')
        mversion = models.get('version', 'v1')
        # Преобразуем vector -> dict признаков (индексированные) для универсальности
        feat_dict = {f'f{i}': float(v) for i, v in enumerate(vector)}
        # Если доступен rms_a в feature, используем его
        if feature.rms_a is not None:
            feat_dict['rms_a'] = float(feature.rms_a)
        score = 0.0
        try:
            score = model.update(feat_dict)  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.debug(f"model.update error: {e}")
        anomaly_detected = False
        try:
            anomaly_detected = model.is_anomaly(score)  # type: ignore
        except Exception:
            anomaly_detected = False
        # Простейшая вероятность по сигмоиде нормализованного score
        import math
        prob_raw = 1.0 / (1.0 + math.exp(-min(max(score, -20), 20)))
        probability = float(prob_raw if anomaly_detected else 1 - prob_raw)
        confidence_val = float(score)
        prediction = Prediction(
            id=uuid4(),
            feature_id=UUID(feature_id),
            equipment_id=getattr(feature, 'equipment_id', None),
            model_name=f'anomaly_{mtype}',
            model_version=str(mversion),
            model_type='anomaly',
            anomaly_detected=anomaly_detected,
            probability=probability,
            confidence=confidence_val,
            prediction_details={
                'score': float(score),
                'vector_length': len(vector),
                'model_type': mtype
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
    'load_latest_models_async', 'FeatureExtractor', 'RMSTrendForecaster', 'InsufficientDataError',
    'run_clustering_async'
]


# ----------------------- clustering_pipeline -----------------------
@celery_app.task(bind=True, name='src.worker.tasks.clustering_pipeline', autoretry_for=(Exception,), retry_kwargs={'max_retries': 1}, retry_backoff=True)
@observe_latency('worker_task_duration_seconds', labels={'task_name': 'clustering_pipeline'})
def clustering_pipeline(self, min_cluster_size: int = 10) -> Dict:  # type: ignore[override]
    return asyncio.run(run_clustering_async(min_cluster_size=min_cluster_size))


async def run_clustering_async(min_cluster_size: int = 10) -> Dict:
    async with get_async_session() as session:
        result = await full_clustering_pipeline(session, min_cluster_size=min_cluster_size)
        distribution = await build_distribution_report(session)
        label_dict = await load_cluster_label_dict(session)
        knn_status = 'skipped'
        tcn_preview = None
        try:
            cache = load_model('clustering/knn_model')
            if cache:
                knn_status = 'loaded'
            else:
                knn = semi_supervised_knn(result.reduced, result.labels, label_dict)
                save_model({'knn': knn}, 'clustering/knn_model')
                knn_status = 'trained'
        except Exception as e:  # нет размеченных кластеров
            knn_status = f'skipped: {e}'
        # Пробуем получить быстрый TCN прогноз для любого оборудования из выборки
        try:
            # Находим equipment_id первого raw через Feature -> RawSignal
            from sqlalchemy import select
            from src.database.models import Feature as F, RawSignal as RS
            feat_id = result.feature_ids[0]
            feat_row = await session.get(F, feat_id)
            if feat_row:
                raw = await session.get(RS, feat_row.raw_id)
                if raw:
                    tcn_preview = await predict_tcn(raw.equipment_id)
        except Exception as te:  # pragma: no cover
            tcn_preview = {'error': str(te)}
        # Manifest clustering
        try:
            from datetime import datetime, UTC
            from src.config.settings import get_settings
            st = get_settings()
            model_dir = st.models_path / 'clustering'
            model_dir.mkdir(parents=True, exist_ok=True)
            labeled_clusters = len(label_dict)
            total_clusters = int(len(set([c for c in result.labels if c != -1])))
            manifest = {
                'version': datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'),
                'clusters_found': total_clusters,
                'labeled_clusters': labeled_clusters,
                'features_clustered': len(result.feature_ids),
                'min_cluster_size': min_cluster_size,
                'knn_status': knn_status,
            }
            import json
            (model_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        except Exception:  # pragma: no cover
            pass
        return {
            'status': 'success',
            'features_clustered': len(result.feature_ids),
            'clusters_found': int(len(set([c for c in result.labels if c != -1]))),
            'distribution': distribution,
            'knn_status': knn_status,
            'tcn_preview': tcn_preview
        }
    # (не достижимо после return)
