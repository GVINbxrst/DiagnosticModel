"""Ensure monthly partitions exist for raw_signals & features.

Usage (inside container or venv):
  python scripts/ensure_partitions.py --months-back 12 --months-forward 6

Idempotent: creates missing monthly partitions using dynamic SQL. Safe to run periodically (e.g. daily cron or Celery beat).
"""
from __future__ import annotations
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sqlalchemy import text
import asyncio

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

PARTITION_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table}_{suffix}
    PARTITION OF {table}
    FOR VALUES FROM (%(start)s) TO (%(end)s);
"""

async def ensure_range(table: str, column: str, start_month: datetime, end_month: datetime):
    async with get_async_session() as session:  # type: ignore
        cur = start_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        stop = end_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while cur <= stop:
            nxt = cur + relativedelta(months=1)
            suffix = cur.strftime('%Y_%m')
            sql = PARTITION_TEMPLATE.format(table=table, suffix=suffix)
            try:
                await session.execute(text(sql), {
                    'start': cur.isoformat(),
                    'end': nxt.isoformat()
                })
                logger.info(f"partition ok {table}_{suffix}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"partition fail {table}_{suffix}: {e}")
            cur = nxt
        await session.commit()

async def main(months_back: int, months_forward: int):
    now = datetime.utcnow()
    start = now - relativedelta(months=months_back)
    end = now + relativedelta(months=months_forward)
    await ensure_range('raw_signals','recorded_at', start, end)
    await ensure_range('features','window_start', start, end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--months-back', type=int, default=12)
    parser.add_argument('--months-forward', type=int, default=6)
    args = parser.parse_args()
    asyncio.run(main(args.months_back, args.months_forward))
