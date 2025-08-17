"""Простейший мониторинг дрейфа кластерного распределения.

Сохраняет предыдущое распределение кластеров в models/clustering/drift_baseline.json
и при вызове compute_and_update_drift пересчитывает JS divergence между прошлым и текущим
распределением по размерам кластеров.
"""
from __future__ import annotations
import json, os, math
from pathlib import Path
from typing import Dict, Any
from src.config.settings import get_settings
from src.utils.metrics import set_gauge


def _js_divergence(p: Dict[int,int], q: Dict[int,int]) -> float:
    # Преобразуем в вероятностные распределения по общей унии кластеров
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    sp = sum(p.values()) or 1
    sq = sum(q.values()) or 1
    def prob(dist, k):
        return (dist.get(k,0) / (sp if dist is p else sq))
    m = {k: 0.5*(prob(p,k)+prob(q,k)) for k in keys}
    def kld(a,b):
        eps = 1e-12
        v = 0.0
        for k in keys:
            pa = prob(a,k) + eps
            pb = prob(b,k) + eps
            v += pa * math.log(pa/pb)
        return v
    js = 0.5 * kld(p,m) + 0.5 * kld(q,m)
    # Нормализуем в 0..1 грубо: JS уже ограничена логарифмом базы e; приведём через (1 - exp(-js)) как сглаженную шкалу
    return min(1.0, 1 - math.exp(-js))


def compute_and_update_drift(current_distribution: Dict[str, Any]) -> float:
    st = get_settings()
    base_dir = st.models_path / 'clustering'
    os.makedirs(base_dir, exist_ok=True)
    baseline_path = base_dir / 'drift_baseline.json'
    # Текущее распределение в dict cluster_id->count (исключая None)
    clusters = {int(c['cluster_id']): int(c['count']) for c in current_distribution.get('clusters', []) if c.get('cluster_id') is not None}
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps({'clusters': clusters}, ensure_ascii=False, indent=2), encoding='utf-8')
        set_gauge('clustering_drift_score', 0.0)
        return 0.0
    try:
        prev = json.loads(baseline_path.read_text(encoding='utf-8')).get('clusters', {})
    except Exception:
        prev = {}
    drift = _js_divergence(prev, clusters)
    set_gauge('clustering_drift_score', drift)
    # Обновляем baseline (скользящее окно можно позже добавить)
    baseline_path.write_text(json.dumps({'clusters': clusters}, ensure_ascii=False, indent=2), encoding='utf-8')
    return drift

__all__ = ['compute_and_update_drift']
