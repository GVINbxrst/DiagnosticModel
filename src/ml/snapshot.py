"""Создание snapshot обучающего набора (features + embeddings + дефектные метки).

Сохраняет parquet и manifest в models/snapshots/<timestamp>/dataset.parquet.
Использует только фичи с embedding и ненулевыми cluster_id. Добавляет колонку defect_label если есть
в ClusterLabel.
"""
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import json, os
from typing import List, Dict, Any
from src.database.models import Feature, ClusterLabel
from src.config.settings import get_settings
from src.utils.metrics import set_gauge

async def build_training_snapshot(session: AsyncSession) -> Dict[str, Any]:
    # Выбираем нужные поля
    res = await session.execute(select(Feature.id, Feature.cluster_id, Feature.extra))
    rows = res.all()
    if not rows:
        return {'status': 'empty'}
    # Метки кластеров
    lab_res = await session.execute(select(ClusterLabel.cluster_id, ClusterLabel.defect_id))
    label_map = {cid: defect for cid, defect in lab_res.all()}
    records: List[Dict[str, Any]] = []
    for fid, cid, extra in rows:
        if not extra:
            continue
        emb = extra.get('embedding') if isinstance(extra, dict) else None
        if not emb or not isinstance(emb, list):
            continue
        records.append({
            'feature_id': str(fid),
            'cluster_id': cid,
            'embedding': emb,
            'embedding_dim': len(emb),
            'defect_label': label_map.get(cid) if cid is not None else None
        })
    if not records:
        return {'status': 'no_embeddings'}
    df = pd.DataFrame(records)
    st = get_settings()
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    out_dir = st.models_path / 'snapshots' / ts
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = out_dir / 'dataset.parquet'
    df.to_parquet(parquet_path, index=False)
    manifest = {
        'created_at': datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        'records': len(df),
        'clusters': int(df['cluster_id'].nunique() if 'cluster_id' in df else 0),
        'labeled_clusters': int(df.dropna(subset=['defect_label'])['cluster_id'].nunique() if 'defect_label' in df else 0),
        'path': str(parquet_path)
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    set_gauge('training_snapshot_timestamp', datetime.utcnow().timestamp())
    return {'status': 'ok', 'snapshot_dir': str(out_dir), **manifest}

__all__ = ['build_training_snapshot']
