#!/usr/bin/env python
"""Полный E2E пайплайн:
1. (Опц.) Инжест локальных CSV -> RawSignal
2. Обработка RawSignal -> Feature (windowed)
3. Обучение модели (full / fallback isolation forest)
4. Детекция аномалий (новые Feature без Prediction)
5. Прогноз трендов по активному оборудованию
6. Отчёт (counts)

Usage:
  python scripts/e2e_pipeline.py --csv data/raw --limit 100 --train-full
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, List, Optional

from sqlalchemy import select

from src.database.connection import get_async_session
from src.database.models import (
    RawSignal, Feature, Prediction, Equipment, EquipmentStatus, ProcessingStatus, EquipmentType
)
from src.worker.tasks import _process_raw_async, _detect_anomalies_async, _forecast_trend_async
from src.ml.train import train_anomaly_models  # type: ignore
from src.utils.logger import get_logger
from src.data_processing.csv_loader import CSVLoader

logger = get_logger(__name__)


# ---------------- Model -----------------
async def ensure_model(full: bool = False) -> None:
    """Упрощённый ensure_model: выполняет обучение актуальных anomaly моделей один раз.

    Legacy IsolationForest удалён; вызываем только train_anomaly_models при full=True.
    """
    if full:
        try:
            await train_anomaly_models(output_dir=None)
            logger.info("Актуальные anomaly модели обучены")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Обучение anomaly моделей не удалось: {e}")
    else:
        logger.info("full флаг не установлен – пропуск обучения (ожидается потоковая модель)")


# --------------- Queries -----------------
async def collect_unprocessed(limit: Optional[int] = None) -> List[str]:
    async with get_async_session() as session:
        q = select(RawSignal.id).where(RawSignal.processing_status != ProcessingStatus.COMPLETED)
        if limit:
            q = q.limit(limit)
        res = await session.execute(q)
        return [str(r[0]) for r in res.fetchall()]


async def collect_features_without_pred(limit: Optional[int] = None) -> List[str]:
    async with get_async_session() as session:
        sub = select(Prediction.feature_id)
        q = select(Feature.id).where(~Feature.id.in_(sub))
        if limit:
            q = q.limit(limit)
        res = await session.execute(q)
        return [str(r[0]) for r in res.fetchall()]


async def active_equipment(limit: Optional[int] = None) -> List[str]:
    async with get_async_session() as session:
        q = select(Equipment.id).where(Equipment.status == EquipmentStatus.ACTIVE)
        if limit:
            q = q.limit(limit)
        res = await session.execute(q)
        return [str(r[0]) for r in res.fetchall()]


# --------------- Ingest ------------------
async def ingest_csv_directory(csv_dir: str, pattern: str, limit_files: Optional[int]) -> int:
    path = Path(csv_dir)
    if not path.exists():
        logger.warning(f"Каталог CSV не найден: {csv_dir}")
        return 0
    files = sorted(path.rglob(pattern)) if '**' in pattern else sorted(path.glob(pattern))
    if limit_files:
        files = files[:limit_files]
    if not files:
        logger.info("CSV файлы не найдены — пропуск инжеста")
        return 0
    logger.info(f"Найдено файлов для инжеста: {len(files)}")
    loader = CSVLoader()
    total_rows = 0
    for idx, f in enumerate(files, 1):
        try:
            stats = await loader.load_csv_file(f)
            stats.finish()
            total_rows += stats.processed_rows
            logger.info(f"[{idx}/{len(files)}] {f.name}: rows={stats.processed_rows} raw={len(stats.raw_signal_ids)}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ошибка инжеста {f}: {e}")
    logger.info(f"Инжест завершён: total_rows={total_rows}")
    return total_rows


async def ensure_default_equipment() -> None:
    """Гарантировать наличие хотя бы одного активного оборудования.

    Если отсутствует активное оборудование – создаём Synthetic Motor 001.
    Это упрощает прогноз и дальнейшие шаги без ручного ввода сущности.
    """
    async with get_async_session() as session:
        from sqlalchemy import select
        res = await session.execute(select(Equipment).where(Equipment.status == EquipmentStatus.ACTIVE))
        existing = res.scalars().first()
        if existing:
            return
        # Нет активного – создаём
        eq = Equipment(
            equipment_id="MOTOR_SYN_001",
            name="Synthetic Motor 001",
            type=EquipmentType.INDUCTION_MOTOR,
            status=EquipmentStatus.ACTIVE,
            specifications={"created_auto": True}
        )
        session.add(eq)
        await session.commit()
        logger.info("Создано синтетическое оборудование MOTOR_SYN_001 (ACTIVE)")


# --------------- Pipeline ----------------
async def run_pipeline(csv_dir: Optional[str], limit: Optional[int], train_full: bool, pattern: str, max_files: Optional[int]):
    # 0. Ensure equipment
    await ensure_default_equipment()
    # 1. Инжест CSV (если указан каталог)
    if csv_dir:
        await ingest_csv_directory(csv_dir, pattern, max_files)

    # 2. Обработка RawSignal -> Feature
    raw_ids = await collect_unprocessed(limit)
    logger.info(f"RawSignal для обработки: {len(raw_ids)}")
    for rid in raw_ids:
        try:
            await _process_raw_async(rid)
        except Exception as e:  # pragma: no cover
            logger.warning(f"process_raw {rid}: {e}")

    # 3. Обучение / загрузка модели
    await ensure_model(full=train_full)

    # 4. Детекция аномалий
    feat_ids = await collect_features_without_pred(limit)
    logger.info(f"Features без Prediction: {len(feat_ids)}")
    for fid in feat_ids:
        try:
            await _detect_anomalies_async(fid)
        except Exception as e:  # pragma: no cover
            logger.warning(f"detect_anomalies {fid}: {e}")

    # 5. Прогноз трендов
    eq_ids = await active_equipment(limit)
    logger.info(f"Оборудование для прогноза: {len(eq_ids)}")
    for eid in eq_ids:
        try:
            await _forecast_trend_async(eid)
        except Exception as e:  # pragma: no cover
            logger.warning(f"forecast {eid}: {e}")

    # 6. Отчёт
    from scripts.analytics_report import build_report  # локальный импорт
    report = await build_report()
    print("\n=== E2E PIPELINE DONE ===")
    for k, v in report.items():
        print(f"{k}: {v}")
    logger.info(f"Итоговый отчёт: {report}")
    # 6b. HTML отчёт (если скрипт доступен)
    try:
        from scripts.build_html_report import build_html_report
        html_path = await build_html_report(output_dir="reports")
        logger.info(f"HTML отчёт сохранён: {html_path}")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Не удалось построить HTML отчёт: {e}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, help='Каталог CSV для инжеста (опционально)')
    ap.add_argument('--pattern', default='*.csv', help='Шаблон файлов (по умолчанию *.csv)')
    ap.add_argument('--max-files', type=int, help='Максимум CSV файлов для инжеста')
    ap.add_argument('--limit', type=int, help='Лимит сущностей на обработку (raw/features/equipment)')
    ap.add_argument('--train-full', action='store_true', help='Полное обучение (IsolationForest+DBSCAN) вместо fallback')
    args = ap.parse_args()
    asyncio.run(run_pipeline(args.csv, args.limit, args.train_full, args.pattern, args.max_files))
