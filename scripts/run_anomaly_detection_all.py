#!/usr/bin/env python
"""Прогон детекции аномалий для последних признаков без Prediction.

Usage:
  python scripts/run_anomaly_detection_all.py --limit 100
"""
import asyncio
from sqlalchemy import select
from uuid import UUID

from src.database.connection import get_async_session
from src.database.models import Feature, Prediction
from src.worker.tasks import _detect_anomalies_async
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def run_detection(limit: int | None = None):
    async with get_async_session() as session:
        subq = select(Prediction.feature_id)
        q = select(Feature.id).where(~Feature.id.in_(subq))
        if limit:
            q = q.limit(limit)
        result = await session.execute(q)
        feature_ids = [str(r[0]) for r in result.fetchall()]
    logger.info(f"Найдено {len(feature_ids)} признаков без предсказаний")
    done = 0
    for fid in feature_ids:
        try:
            res = await _detect_anomalies_async(fid)
            if res.get('status') == 'success':
                done += 1
        except Exception as e:
            logger.warning(f"Ошибка аномалий feature {fid}: {e}")
    logger.info(f"Детекция завершена: {done}/{len(feature_ids)} success")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    asyncio.run(run_detection(args.limit))
