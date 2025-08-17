"""Вычисление метрик готовности обучающих данных.

collect_readiness_metrics(session) подсчитывает показатели и
обновляет Prometheus gauge метрики.
"""
from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from statistics import median
from typing import Dict, Any, List
from datetime import datetime, timezone

from src.database.models import Feature, ClusterLabel
from src.utils.metrics import set_gauge

async def collect_readiness_metrics(session: AsyncSession) -> Dict[str, Any]:
    total_q = await session.execute(select(func.count(Feature.id)))
    total_features = int(total_q.scalar() or 0)

    # Количество с embedding (используем JSONB оператор -> 'embedding')
    emb_q = await session.execute(select(func.count(Feature.id)).where(Feature.extra[('embedding')].isnot(None)))  # type: ignore
    with_embedding = int(emb_q.scalar() or 0)
    coverage = (with_embedding / total_features) if total_features else 0.0

    dist_q = await session.execute(select(Feature.cluster_id, func.count(Feature.id)).group_by(Feature.cluster_id))
    rows = dist_q.all()
    noise_count = 0
    cluster_sizes: List[int] = []
    for cid, cnt in rows:
        if cid is None:
            noise_count += cnt
        else:
            cluster_sizes.append(cnt)
    noise_ratio = (noise_count / total_features) if total_features else 0.0

    labeled_q = await session.execute(select(func.count(func.distinct(ClusterLabel.cluster_id))))
    labeled_clusters = int(labeled_q.scalar() or 0)
    total_clusters = len(cluster_sizes)
    label_coverage = (labeled_clusters / total_clusters) if total_clusters else 0.0

    if cluster_sizes:
        sorted_sizes = sorted(cluster_sizes)
        p50 = median(sorted_sizes)
        p90_index = max(0, int(0.9 * len(sorted_sizes)) - 1)
        p90 = sorted_sizes[p90_index]
        min_labeled_size = 0
        if labeled_clusters:
            labeled_sizes = [cnt for cid, cnt in rows if cid is not None][:labeled_clusters]
            if labeled_sizes:
                min_labeled_size = min(labeled_sizes)
    else:
        p50 = p90 = min_labeled_size = 0

    max_time_q = await session.execute(select(func.max(Feature.window_end)))
    max_time = max_time_q.scalar()
    if max_time is not None:
        if max_time.tzinfo is None:
            max_time = max_time.replace(tzinfo=timezone.utc)
        recency = (datetime.now(timezone.utc) - max_time).total_seconds()
    else:
        recency = 0.0

    set_gauge('features_with_embedding_total', with_embedding)
    set_gauge('embedding_coverage_ratio', coverage)
    set_gauge('clustering_noise_ratio', noise_ratio)
    set_gauge('clustering_label_coverage_ratio', label_coverage)
    set_gauge('min_cluster_size_labeled', min_labeled_size)
    set_gauge('p50_cluster_size', float(p50))
    set_gauge('p90_cluster_size', float(p90))
    set_gauge('data_recency_seconds', recency)

    return {
        'total_features': total_features,
        'with_embedding': with_embedding,
        'embedding_coverage': coverage,
        'noise_ratio': noise_ratio,
        'labeled_clusters': labeled_clusters,
        'total_clusters': total_clusters,
        'label_coverage': label_coverage,
        'min_labeled_cluster_size': min_labeled_size,
        'p50_cluster_size': p50,
        'p90_cluster_size': p90,
        'data_recency_seconds': recency
    }

__all__ = ['collect_readiness_metrics']
