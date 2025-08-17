"""Clustering & semi-supervised classification over feature embeddings.

Pipeline:
1. load_embeddings(session) -> np.ndarray, feature_ids
2. reduce_embeddings(X) -> X_low (UMAP)
3. cluster_embeddings(X_low) -> labels, clusterer
4. persist_cluster_ids(session, feature_ids, labels)
5. build_distribution_report(session) -> dict
6. load_cluster_label_dict(session) -> mapping cluster_id -> defect_label
7. semi_supervised_classifier(X_low, labels, label_dict) -> fitted kNN model

Notes:
- Unclustered points in HDBSCAN are labeled as -1 (noise). We leave cluster_id NULL for noise.
- We store cluster_id on Feature.cluster_id.
- Manual mapping cluster_id->defect_label stored in ClusterLabel table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:  # Optional heavy deps
    import umap
except ImportError:  # pragma: no cover
    umap = None  # type: ignore
try:
    import hdbscan
except ImportError:  # pragma: no cover
    hdbscan = None  # type: ignore

from sklearn.neighbors import KNeighborsClassifier
from .utils import save_model, load_model
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Feature, ClusterLabel


@dataclass
class ClusteringResult:
    feature_ids: List[str]
    embeddings: np.ndarray
    reduced: np.ndarray
    labels: np.ndarray
    clusterer: Any


def extract_embeddings(feature: Feature) -> Optional[List[float]]:
    if not feature.extra:
        return None
    emb = feature.extra.get("embedding") if isinstance(feature.extra, dict) else None
    if emb is None:
        return None
    if not isinstance(emb, (list, tuple)):
        return None
    # ensure numeric
    try:
        return [float(x) for x in emb]
    except Exception:
        return None


async def load_embeddings(session: AsyncSession) -> Tuple[np.ndarray, List[str]]:
    q = await session.execute(select(Feature.id, Feature.extra))
    rows = q.all()
    feature_ids: List[str] = []
    vectors: List[List[float]] = []
    for fid, extra in rows:
        if not extra or "embedding" not in extra:
            continue
        emb = extra.get("embedding")
        if isinstance(emb, list) and emb:
            try:
                vectors.append([float(x) for x in emb])
                feature_ids.append(str(fid))
            except Exception:  # skip malformed
                continue
    if not vectors:
        raise ValueError("No embeddings found to cluster.")
    return np.asarray(vectors, dtype=np.float32), feature_ids


def reduce_embeddings(X: np.ndarray, n_components: int = 2, random_state: int = 42):
    if umap is None:
        raise ImportError("umap-learn is not installed. Add 'umap-learn' to requirements.")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, metric="euclidean")
    X_low = reducer.fit_transform(X)
    return X_low, reducer


def cluster_embeddings(X_low: np.ndarray, min_cluster_size: int = 10, min_samples: Optional[int] = None):
    if hdbscan is None:
        raise ImportError("hdbscan is not installed. Add 'hdbscan' to requirements.")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples or min_cluster_size // 2,
                                metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(X_low)
    return labels, clusterer


async def persist_cluster_ids(session: AsyncSession, feature_ids: List[str], labels: np.ndarray) -> int:
    updated = 0
    for fid, label in zip(feature_ids, labels):
        if label == -1:
            continue  # noise -> keep NULL
        feat: Feature | None = await session.get(Feature, fid)
        if feat:
            feat.cluster_id = int(label)
            updated += 1
    await session.commit()
    return updated


async def build_distribution_report(session: AsyncSession) -> Dict[str, Any]:
    from sqlalchemy import func
    q = await session.execute(
        select(Feature.cluster_id, func.count(Feature.id)).group_by(Feature.cluster_id)
    )
    rows = q.all()
    total = sum(r[1] for r in rows)
    distribution = []
    for cid, count in rows:
        pct = (count / total * 100.0) if total else 0.0
        distribution.append({"cluster_id": cid, "count": count, "percent": round(pct, 2)})
    return {"total": total, "clusters": distribution}


async def load_cluster_label_dict(session: AsyncSession) -> Dict[int, str]:
    q = await session.execute(select(ClusterLabel.cluster_id, ClusterLabel.defect_id))
    return {cid: defect_id for cid, defect_id in q.all()}


def semi_supervised_knn(X_low: np.ndarray, labels: np.ndarray, cluster_label_map: Dict[int, str], k: int = 5):
    # Build training set from points whose cluster has a label mapping.
    y: List[str] = []
    X_train: List[np.ndarray] = []
    for i, c in enumerate(labels):
        if c == -1:
            continue
        if c in cluster_label_map:
            y.append(cluster_label_map[c])
            X_train.append(X_low[i])
    if not X_train:
        raise ValueError("No labeled clusters to train kNN.")
    X_train_arr = np.vstack(X_train)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(X_train_arr)))
    knn.fit(X_train_arr, y)
    return knn


async def full_clustering_pipeline(session: AsyncSession, min_cluster_size: int = 10, n_components: int = 2) -> ClusteringResult:
    # Попытка загрузить кеш
    cached = load_model("clustering_pipeline_cache")
    if cached:
        try:
            feature_ids = cached["feature_ids"]
            X = cached["X"]
            X_low = cached["X_low"]
            labels = cached["labels"]
            clusterer = cached["clusterer"]
            return ClusteringResult(feature_ids, X, X_low, labels, clusterer)
        except Exception:
            pass
    X, feature_ids = await load_embeddings(session)
    X_low, _ = reduce_embeddings(X, n_components=n_components)
    labels, clusterer = cluster_embeddings(X_low, min_cluster_size=min_cluster_size)
    await persist_cluster_ids(session, feature_ids, labels)
    save_model({"feature_ids": feature_ids, "X": X, "X_low": X_low, "labels": labels, "clusterer": clusterer}, "clustering_pipeline_cache")
    return ClusteringResult(feature_ids, X, X_low, labels, clusterer)
