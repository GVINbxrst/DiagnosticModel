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

@celery_app.task(name='maintenance.ensure_partitions')
def ensure_partitions_task(months_back: int = 12, months_forward: int = 6):
    """Периодическая задача расширения партиций (идемпотентна)."""
    import asyncio
    from scripts.ensure_partitions import main as run_main  # type: ignore
    print(f"[Celery] ensure_partitions start back={months_back} forward={months_forward}")
    asyncio.run(run_main(months_back, months_forward))
    return {"status": "ok"}

@celery_app.task(name='health.ping')
def health_ping():
    return {"status": "ok"}

@celery_app.on_after_configure.connect  # pragma: no cover
def setup_periodic_tasks(sender, **kwargs):
    # Ежедневное обеспечение партиций
    sender.add_periodic_task(24 * 3600, ensure_partitions_task.s(12, 6), name='ensure_partitions_daily')
    # Ежедневное переобучение моделей аномалий (в ночное окно ~02:00 UTC)
    sender.add_periodic_task(24 * 3600, retrain_models_task.s(), name='retrain_models_daily', expires=3600)
    # Периодический прогноз (каждые 6 часов)
    sender.add_periodic_task(6 * 3600, forecast_all_equipment_task.s(), name='forecast_all_equipment_6h', expires=1800)
    # Кластеризация (каждые 12 часов)
    sender.add_periodic_task(12 * 3600, clustering_pipeline_task.s(), name='clustering_pipeline_12h', expires=3600)
    # Обновление kNN по кластерам (каждые 6 часов)
    sender.add_periodic_task(6 * 3600, retrain_knn_task.s(), name='retrain_knn_6h', expires=1800)

@celery_app.task(name='ml.retrain_models')
def retrain_models_task():
    """Периодическая задача переобучения моделей аномалий.

    Выполняет:
      - Загрузку последних признаков
      - Обучение IsolationForest + scaler
      - Сохранение в models/anomaly_detection/latest
      - Обновление метрик (model_retrain_total, model_version_info)
    """
    import traceback, os, time
    from src.config.settings import get_settings
    from src.ml.train import retrain_stream_or_stats_minimal
    from src.utils.metrics import increment_counter, set_gauge, model_retrain_total, model_version_gauge
    st = get_settings()
    t0 = time.time()
    try:
        # Минимальное обучение (используем helper из train.py или fallback)
        version = retrain_stream_or_stats_minimal()
        # Читаем manifest для доп. сведений
        st_models = st.models_path / 'anomaly_detection' / 'latest' / 'manifest.json'
        model_type = None
        threshold = None
        if st_models.exists():
            try:
                import json
                man = json.loads(st_models.read_text(encoding='utf-8'))
                model_type = man.get('model_type')
                threshold = man.get('threshold')
                p98 = man.get('scores_p98')
                outlier_ratio = man.get('outlier_ratio')
                try:
                    from src.utils.metrics import anomaly_retrain_score_p98, anomaly_retrain_outlier_ratio
                    if p98 is not None:
                        anomaly_retrain_score_p98.labels(model_type=model_type or 'unknown').set(float(p98))
                    if outlier_ratio is not None:
                        anomaly_retrain_outlier_ratio.labels(model_type=model_type or 'unknown').set(float(outlier_ratio))
                except Exception:
                    pass
            except Exception:
                pass
        model_retrain_total.labels(model_group='anomaly', status='success').inc()
        model_version_gauge.labels(model_group='anomaly', version=str(version)).set(1)
        # GC старых версий (best effort)
        try:
            from src.ml.gc_models import gc_anomaly_latest
            gc_anomaly_latest()
        except Exception:
            pass
        return {'status': 'success', 'model_version': version, 'model_type': model_type, 'threshold': threshold, 'duration_sec': round(time.time()-t0,2)}
    except Exception as e:  # pragma: no cover
        model_retrain_total.labels(model_group='anomaly', status='failed').inc()
        return {'status': 'failed', 'error': str(e), 'trace': traceback.format_exc()}

@celery_app.task(name='forecast.generate_all')
def forecast_all_equipment_task(limit: int = 50):
    """Фоновый прогноз для активного оборудования.

    Выбирает до limit активных устройств и генерирует/обновляет прогноз RMS (все фазы) через RMSTrendForecaster.
    Сохраняет запись в Forecast и дублирует в Prediction (aggregated) для быстрой выборки.
    """
    import time, traceback
    from uuid import UUID as _UUID
    from sqlalchemy import select
    from src.database.connection import get_async_session
    from src.database.models import Equipment, Forecast, Prediction
    from src.utils.metrics import forecast_tasks_total, forecasts_generated_total
    from src.ml.forecasting import RMSTrendForecaster
    t0 = time.time()
    ok = 0
    failed = 0
    try:
        async def _run():
            nonlocal ok, failed
            async with get_async_session() as session:
                res = await session.execute(select(Equipment).where(Equipment.status == 'active').limit(limit))
                equipments = res.scalars().all()
                for eq in equipments:
                    try:
                        forecaster = RMSTrendForecaster()
                        result = await forecaster.forecast_equipment_trends(eq.id)
                        summary = result.get('summary', {})
                        max_prob = summary.get('max_anomaly_probability', 0.0)
                        fc = Forecast(
                            equipment_id=eq.id,
                            horizon=result.get('forecast_steps', 24),
                            method='auto',
                            forecast_data=result,
                            probability_over_threshold=max_prob,
                            model_version='v1'
                        )
                        session.add(fc)
                        # Сводная Prediction запись (без feature_id)
                        pd = Prediction(
                            feature_id=_UUID(int=0),  # заглушка
                            equipment_id=eq.id,
                            model_name='rms_trend_forecasting',
                            model_version='v1',
                            model_type='forecast',
                            anomaly_detected=bool(max_prob and max_prob > 0.7),
                            probability=1 - float(max_prob or 0.0),
                            confidence=float(max_prob or 0.0),
                            prediction_details={'summary': summary},
                            created_at=None  # авто
                        )
                        session.add(pd)
                        await session.commit()
                        ok += 1
                        forecasts_generated_total.labels(model_name='rms_trend_forecasting', equipment_id=str(eq.id), status='success').inc()
                    except Exception as ie:  # pragma: no cover
                        failed += 1
                        await session.rollback()
                        forecasts_generated_total.labels(model_name='rms_trend_forecasting', equipment_id=str(eq.id), status='failed').inc()
        import asyncio
        asyncio.run(_run())
        forecast_tasks_total.labels(task='forecast_all_equipment', status='success').inc()
        return {'status': 'success', 'processed': ok, 'failed': failed, 'duration_sec': round(time.time()-t0,2)}
    except Exception as e:  # pragma: no cover
        forecast_tasks_total.labels(task='forecast_all_equipment', status='failed').inc()
        return {'status': 'failed', 'error': str(e), 'trace': traceback.format_exc()}

@celery_app.task(name='clustering.run_periodic')
def clustering_pipeline_task(min_cluster_size: int = 10):
    """Периодический запуск кластеризации (тонкая обёртка над async пайплайном)."""
    import time, traceback
    import asyncio
    from src.utils.metrics import clustering_runs_total, clustering_last_duration_seconds
    from src.worker.tasks import run_clustering_async
    t0 = time.time()
    try:
        res = asyncio.run(run_clustering_async(min_cluster_size=min_cluster_size))
        # drift metric
        try:
            from src.api.routes.admin_clustering import build_distribution_report  # may cause circular; fallback manual
        except Exception:
            build_distribution_report = None
        try:
            from src.database.connection import get_async_session
            from src.ml.drift_monitor import compute_and_update_drift
            if build_distribution_report:
                async def _dist():
                    async with get_async_session() as s:
                        from src.ml.clustering import build_distribution_report as dist
                        return await dist(s)
                import asyncio as _a
                dist_data = _a.run(_dist())
                compute_and_update_drift(dist_data)
        except Exception:
            pass
        clustering_runs_total.labels(status='success').inc()
        clustering_last_duration_seconds.set(time.time()-t0)
        return res
    except Exception as e:  # pragma: no cover
        clustering_runs_total.labels(status='failed').inc()
        return {'status': 'failed', 'error': str(e), 'trace': traceback.format_exc()}

@celery_app.task(name='clustering.retrain_knn')
def retrain_knn_task():
    """Периодическое переобучение semi-supervised kNN если есть размеченные кластеры.

    Условие: хотя бы 1 кластер размечен. В будущем можно добавить триггер по приросту coverage.
    """
    import asyncio, traceback, json, time, os
    from src.database.connection import get_async_session
    from src.ml.clustering import load_embeddings, reduce_embeddings, cluster_embeddings, load_cluster_label_dict, semi_supervised_knn
    from src.config.settings import get_settings
    t0 = time.time()
    try:
        async def _run():
            async with get_async_session() as session:
                X, _ = await load_embeddings(session)
                X_low, _ = reduce_embeddings(X)
                labels, _ = cluster_embeddings(X_low)
                label_map = await load_cluster_label_dict(session)
                if not label_map:
                    return {'status': 'skipped', 'reason': 'no_labeled_clusters'}
                knn = semi_supervised_knn(X_low, labels, label_map)
                st = get_settings()
                out_dir = st.models_path / 'clustering'
                os.makedirs(out_dir, exist_ok=True)
                import joblib
                joblib.dump({'knn': knn}, out_dir / 'knn_model.pkl')
                # manifest lite
                manifest = {
                    'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'labeled_clusters': len(label_map),
                    'coverage_ratio': len(label_map) / max(1, len([c for c in set(labels) if c != -1])),
                    'n_neighbors': getattr(knn, 'n_neighbors', None)
                }
                (out_dir / 'knn_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
                # Snapshot датасета (best-effort)
                try:
                    from src.ml.snapshot import build_training_snapshot
                    snap = await build_training_snapshot(session)
                except Exception:
                    snap = {'snapshot': 'failed'}
                return {'status': 'trained', 'labeled_clusters': len(label_map), 'snapshot': snap.get('status')}
        res = asyncio.run(_run())
        res['duration_sec'] = round(time.time()-t0,2)
        return res
    except Exception as e:  # pragma: no cover
        return {'status': 'failed', 'error': str(e), 'trace': traceback.format_exc()}

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
