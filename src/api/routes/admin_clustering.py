from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.middleware.auth import require_any_role
from src.database.connection import db_session
from src.database.models import ClusterLabel, DefectCatalog
from src.ml.clustering import build_distribution_report, full_clustering_pipeline, load_cluster_label_dict, semi_supervised_knn
from src.ml.data_readiness import collect_readiness_metrics
from src.worker.tasks import clustering_pipeline
from src.config.settings import get_settings
from src.ml.snapshot import build_training_snapshot
import json, os
import asyncio

router = APIRouter(prefix="/admin/clustering", tags=["clustering"])


class ClusterLabelPayload(BaseModel):
    defect_id: str
    description: Optional[str] = None


@router.post("/run")
async def run_clustering(min_cluster_size: int = 10, session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    """Синхронный запуск пайплайна (debug/admin)."""
    result = await full_clustering_pipeline(session, min_cluster_size=min_cluster_size)
    distribution = await build_distribution_report(session)
    return {
        'features_clustered': len(result.feature_ids),
        'clusters_found': int(len(set([c for c in result.labels if c != -1]))),
        'distribution': distribution
    }


@router.post("/enqueue")
async def enqueue_clustering(min_cluster_size: int = 10, current_user=Depends(require_any_role)):
    """Поставить кластеризацию в очередь Celery."""
    task = clustering_pipeline.delay(min_cluster_size=min_cluster_size)  # type: ignore
    return {'task_id': task.id, 'status': 'queued'}


@router.get("/distribution")
async def get_distribution(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    return await build_distribution_report(session)

@router.get("/summary")
async def get_clustering_summary(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    distribution = await build_distribution_report(session)
    # Читаем manifest если есть
    manifest = None
    try:
        from src.config.settings import get_settings
        import json, os
        st = get_settings()
        path = st.models_path / 'clustering' / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        manifest = None
    # Покрытие меток дефектов
    labeled_map = await load_cluster_label_dict(session)
    covered = len(labeled_map)
    total_clusters = len([c for c in distribution.get('clusters', []) if c.get('cluster_id') not in (None, -1)])
    coverage = (covered / total_clusters) if total_clusters else 0.0
    return {
        'distribution': distribution,
        'manifest': manifest,
        'label_coverage': {
            'labeled_clusters': covered,
            'total_clusters': total_clusters,
            'coverage_ratio': round(coverage, 4)
        }
    }

@router.get("/data-readiness")
async def get_data_readiness(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    """Метрики готовности обучающих данных (embedding coverage, noise ratio, label coverage и т.д.)."""
    metrics = await collect_readiness_metrics(session)
    return metrics

@router.post("/snapshot")
async def create_training_snapshot(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    """Явный запуск snapshot обучающего датасета (parquet + manifest)."""
    result = await build_training_snapshot(session)
    return result

@router.get("/embedding-manifest")
async def get_embedding_manifest(current_user=Depends(require_any_role)):
    st = get_settings()
    path = st.models_path / 'embeddings' / 'manifest.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail="Embedding manifest not found")
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read manifest")


@router.post("/clusters/{cluster_id}/label")
async def set_cluster_label(cluster_id: int, payload: ClusterLabelPayload, session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    # Проверяем наличие дефекта в справочнике
    defect = await session.get(DefectCatalog, payload.defect_id)
    if not defect:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Unknown defect_id")
    existing = await session.execute(select(ClusterLabel).where(ClusterLabel.cluster_id == cluster_id))
    row = existing.scalar_one_or_none()
    if row:
        row.defect_id = payload.defect_id
        row.description = payload.description
    else:
        row = ClusterLabel(cluster_id=cluster_id, defect_id=payload.defect_id, description=payload.description)
        session.add(row)
    await session.commit()
    return {'cluster_id': cluster_id, 'defect_id': payload.defect_id}


@router.get("/labels")
async def list_labels(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    res = await session.execute(select(ClusterLabel))
    labels = res.scalars().all()
    return [{'cluster_id': c.cluster_id, 'defect_id': c.defect_id, 'description': c.description} for c in labels]


@router.post("/train-knn")
async def train_knn(session: AsyncSession = Depends(db_session), current_user=Depends(require_any_role)):
    # Подгружаем сохраненную UMAP/кластеры (перезапуск UMAP ради простоты)
    X, feature_ids = await load_embeddings(session)  # type: ignore
    from src.ml.clustering import reduce_embeddings, cluster_embeddings, load_cluster_label_dict, semi_supervised_knn
    X_low, _ = reduce_embeddings(X)
    labels, _ = cluster_embeddings(X_low)
    label_dict = await load_cluster_label_dict(session)
    knn = semi_supervised_knn(X_low, labels, label_dict)
    import joblib, os
    st = get_settings()
    model_dir = st.models_path / 'clustering'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({'knn': knn}, model_dir / 'knn_model.pkl')
    return {'status': 'trained', 'labeled_points': len(knn._fit_X)}
