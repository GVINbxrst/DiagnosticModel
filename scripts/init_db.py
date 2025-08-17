#!/usr/bin/env python
"""Расширенная инициализация БД (без Alembic) для быстрого запуска.

Шаги:
 1. (опц.) DROP существующей схемы через ORM (только таблицы из моделей) при --drop
 2. Создание ORM таблиц (Base.metadata.create_all) — гарантирует соответствие Python моделям
 3. Последовательно выполняются SQL файлы:
    - sql/schema/001_initial.sql
    - sql/schema/002_indexes.sql
    - sql/schema/003_security_audit.sql
    - sql/procedures/001_core_procedures.sql
    - sql/views/001_analysis_views.sql
 4. (опц.) seed начальных данных: sql/seed/001_initial_data.sql (флаг --seed)

Замечания:
 - Файлы могут создавать те же объекты что и ORM (частичное дублирование допустимо на MVP этапе)
 - Для production рекомендуется Alembic; этот скрипт — быстрый bootstrap.
"""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Sequence
import argparse

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.database.connection import engine
from src.database.models import Base
from src.utils.logger import get_logger

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parent.parent

SCHEMA_FILES: Sequence[str] = [
    "sql/schema/001_initial.sql",
    "sql/schema/002_indexes.sql",
    "sql/schema/003_security_audit.sql",
]
PROC_FILES: Sequence[str] = ["sql/procedures/001_core_procedures.sql"]
VIEW_FILES: Sequence[str] = ["sql/views/001_analysis_views.sql"]
SEED_FILES: Sequence[str] = ["sql/seed/001_initial_data.sql"]


async def _exec_sql_file(path: Path) -> None:
    if not path.exists():
        logger.warning(f"SQL файл отсутствует (пропуск): {path}")
        return
    sql_text = path.read_text(encoding='utf-8')
    # Выполняем как единый блок: asyncpg допускает несколько операторов через ';'
    async with engine.begin() as conn:
        try:
            await conn.execute(text(sql_text))
            logger.info(f"Выполнен SQL файл: {path.relative_to(ROOT)}")
        except SQLAlchemyError as e:
            logger.error(f"Ошибка выполнения {path.name}: {e}")
            raise


async def init_schema(drop: bool = False, skip_orm: bool = False):
    async with engine.begin() as conn:
        if not skip_orm:
            if drop:
                logger.warning("DROP всех ORM таблиц (Base.metadata.drop_all)")
                await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            logger.info("ORM таблицы синхронизированы")
        else:
            logger.info("Пропуск ORM create_all (--skip-orm)")


async def apply_idempotent_patches():
    from sqlalchemy import text as _t
    async with engine.begin() as conn:
        # ENUM processing_status
        await conn.execute(_t("""
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'processing_status') THEN
        CREATE TYPE processing_status AS ENUM ('pending','processing','completed','failed');
    END IF;
END$$;"""))
        # raw_signals.processing_status
        await conn.execute(_t("""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns WHERE table_name='raw_signals' AND column_name='processing_status'
    ) THEN
        ALTER TABLE raw_signals ADD COLUMN processing_status processing_status NOT NULL DEFAULT 'pending';
        CREATE INDEX IF NOT EXISTS idx_raw_signals_processing_status ON raw_signals(processing_status, created_at DESC);
    END IF;
END$$;"""))
        # predictions расширение
        await conn.execute(_t("""
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns WHERE table_name='predictions' AND column_name='equipment_id'
    ) THEN
        ALTER TABLE predictions ADD COLUMN equipment_id UUID REFERENCES equipment(id);
        CREATE INDEX IF NOT EXISTS idx_predictions_equipment ON predictions(equipment_id, created_at DESC);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns WHERE table_name='predictions' AND column_name='anomaly_detected'
    ) THEN
        ALTER TABLE predictions ADD COLUMN anomaly_detected BOOLEAN NOT NULL DEFAULT false;
        CREATE INDEX IF NOT EXISTS idx_predictions_anomaly_detected ON predictions(anomaly_detected, confidence DESC, created_at DESC);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns WHERE table_name='predictions' AND column_name='confidence'
    ) THEN
        ALTER TABLE predictions ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0;
        CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence DESC);
    END IF;
END$$;"""))
    logger.info("Идемпотентные патчи применены")


async def full_init(drop: bool, seed: bool, skip_orm: bool, skip_sql: bool, no_patch: bool):
    await init_schema(drop=drop, skip_orm=skip_orm)
    if not skip_sql:
        for rel in SCHEMA_FILES:
            await _exec_sql_file(ROOT / rel)
        for rel in PROC_FILES:
            await _exec_sql_file(ROOT / rel)
        for rel in VIEW_FILES:
            await _exec_sql_file(ROOT / rel)
        if seed:
            for rel in SEED_FILES:
                await _exec_sql_file(ROOT / rel)
    else:
        logger.info("Пропуск выполнения raw SQL файлов (--skip-sql)")
    if not no_patch:
        await apply_idempotent_patches()
    logger.info("Инициализация БД завершена")


async def smoke_check():
    from src.database.connection import check_connection
    ok = await check_connection()
    if not ok:
        raise SystemExit("DB connection failed")


def parse_args():
    ap = argparse.ArgumentParser(description="DiagMod DB init (bootstrap)")
    ap.add_argument('--drop', action='store_true', help='Сделать DROP ORM таблиц перед созданием')
    ap.add_argument('--seed', action='store_true', help='Загрузить начальные данные (seed)')
    ap.add_argument('--skip-orm', action='store_true', help='Пропустить ORM create_all')
    ap.add_argument('--skip-sql', action='store_true', help='Пропустить выполнение SQL файлов')
    ap.add_argument('--no-patch', action='store_true', help='Отключить идемпотентные корректировки (патчи)')
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(smoke_check())
    asyncio.run(full_init(drop=args.drop, seed=args.seed, skip_orm=args.skip_orm, skip_sql=args.skip_sql, no_patch=args.no_patch))
    print("DB init done")
