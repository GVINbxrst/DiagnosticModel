#!/usr/bin/env python
"""Batch обработка всех необработанных RawSignal: извлечение признаков.

Usage (PowerShell):
  python scripts/process_all_rawsignals.py --limit 200
"""
import asyncio
from typing import Optional
from uuid import UUID

from sqlalchemy import select

from src.database.connection import get_async_session
from src.database.models import RawSignal, ProcessingStatus
from src.worker.tasks import _process_raw_async
from src.utils.logger import get_logger

logger = get_logger(__name__)

async def process_batch(limit: Optional[int] = None):
    async with get_async_session() as session:
        q = select(RawSignal.id).where(RawSignal.processing_status != ProcessingStatus.COMPLETED)
        if limit:
            q = q.limit(limit)
        result = await session.execute(q)
        ids = [str(r[0]) for r in result.fetchall()]
    logger.info(f"Найдено {len(ids)} сигналов для обработки")
    processed = 0
    for rid in ids:
        try:
            res = await _process_raw_async(rid)
            if res.get('status') == 'success':
                processed += 1
        except Exception as e:
            logger.warning(f"Ошибка обработки {rid}: {e}")
    logger.info(f"Готово: успешно={processed} / всего={len(ids)}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    asyncio.run(process_batch(args.limit))
