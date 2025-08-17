#!/usr/bin/env python3
"""Загрузка локальных CSV файлов (одноколоночный или 3-колоночный ток) в RawSignal.

Использование:
  python scripts/ingest_local_csv.py --path C:/data/signals --pattern "*.csv" --limit 20
  python scripts/ingest_local_csv.py file1.csv file2.csv

Шаги:
  1. Поиск файлов по списку путей или каталогу с шаблоном.
  2. Асинхронная загрузка через CSVLoader (батчами) в БД.
  3. Вывод сводной статистики и списка RawSignal id.

По умолчанию пытается подобрать Equipment автоматически (см. CSVLoader.find_equipment_by_filename).
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
from typing import List

from src.data_processing.csv_loader import CSVLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collect_files(inputs: List[str], pattern: str | None, recursive: bool) -> List[Path]:
    files: List[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            glob_pat = pattern or '*.csv'
            if recursive:
                files.extend(p.rglob(glob_pat))
            else:
                files.extend(p.glob(glob_pat))
        else:
            logger.warning(f"Путь не найден: {inp}")
    # Удаляем дубликаты, сортируем по имени
    uniq = sorted({f.resolve() for f in files})
    return list(uniq)


async def ingest(files: List[Path], limit: int | None):
    loader = CSVLoader()
    processed_total = 0
    raw_ids: List[str] = []
    for i, f in enumerate(files, 1):
        if limit and processed_total >= limit:
            break
        logger.info(f"[{i}/{len(files)}] Загрузка {f}")
        stats = await loader.load_csv_file(f)
        stats.finish()
        processed_total += stats.processed_rows
        raw_ids.extend([str(r) for r in stats.raw_signal_ids])
        logger.info(f"Файл {f.name}: строк={stats.processed_rows} raw_signals={len(stats.raw_signal_ids)}")
    print("\n=== ИНЖЕСТ ЗАВЕРШЁН ===")
    print(f"Файлов обработано: {min(len(files), i)}")
    print(f"Всего строк: {processed_total}")
    print(f"RawSignal IDs: {', '.join(raw_ids) if raw_ids else '-'}")


def parse_args():
    ap = argparse.ArgumentParser(description="Загрузка локальных CSV файлов в систему")
    ap.add_argument('inputs', nargs='*', help='Файлы или директории')
    ap.add_argument('--path', help='Каталог с CSV')
    ap.add_argument('--pattern', default='*.csv', help='Шаблон поиска (по умолчанию *.csv)')
    ap.add_argument('--recursive', action='store_true', help='Рекурсивный поиск файлов')
    ap.add_argument('--limit', type=int, help='Ограничение количества строк (общий лимит)')
    return ap.parse_args()


def main():
    args = parse_args()
    inputs: List[str] = []
    if args.path:
        inputs.append(args.path)
    inputs.extend(args.inputs)
    if not inputs:
        print('Укажите хотя бы файл или директорию')
        return 1
    files = collect_files(inputs, args.pattern, args.recursive)
    if not files:
        print('Не найдено CSV файлов')
        return 2
    print(f'Найдено файлов: {len(files)}')
    try:
        asyncio.run(ingest(files, args.limit))
    except KeyboardInterrupt:
        print('\nПрервано пользователем')
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
