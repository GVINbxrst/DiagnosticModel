#!/usr/bin/env python
"""Запуск прогнозов RMS трендов для всего активного оборудования.

Usage:
  python scripts/forecast_all_equipment.py --limit 20
"""
import asyncio
from sqlalchemy import select

from src.database.connection import get_async_session
from src.database.models import Equipment, EquipmentStatus
from src.worker.tasks import _forecast_trend_async
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def run_forecasts(limit: int | None = None):
    async with get_async_session() as session:
        q = select(Equipment.id).where(Equipment.status == EquipmentStatus.ACTIVE)
        if limit:
            q = q.limit(limit)
        result = await session.execute(q)
        equipment_ids = [str(r[0]) for r in result.fetchall()]
    logger.info(f"Активное оборудование: {len(equipment_ids)}")
    ok = 0
    for eid in equipment_ids:
        try:
            res = await _forecast_trend_async(eid)
            if res.get('status') == 'success':
                ok += 1
        except Exception as e:
            logger.warning(f"Ошибка прогноза {eid}: {e}")
    logger.info(f"Прогноз завершён: success={ok}/{len(equipment_ids)}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    asyncio.run(run_forecasts(args.limit))
