#!/usr/bin/env python3
"""
Скрипт для загрузки CSV файлов с токовыми сигналами в базу данных DiagMod
Поддерживает загрузку отдельных файлов и массовую обработку директорий
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

from src.data_processing.csv_loader import CSVLoader, load_csv_files_from_directory
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def load_single_file(
    file_path: Path,
    equipment_id: str = None,
    sample_rate: int = 25600,
    batch_size: int = 10000
):
    """Загрузить один CSV файл"""
    try:
        equipment_uuid = UUID(equipment_id) if equipment_id else None

        loader = CSVLoader(batch_size=batch_size)

        logger.info(f"Начинаем загрузку файла: {file_path}")
        stats = await loader.load_csv_file(
            file_path=file_path,
            equipment_id=equipment_uuid,
            sample_rate=sample_rate
        )

        print(f"\n✓ Файл {file_path.name} успешно загружен:")
        print(f"  - Обработано строк: {stats.processed_rows:,}")
        print(f"  - Пачек: {stats.batches_processed}")
        print(f"  - Время: {stats.processing_time:.1f} сек")
        print(f"  - Скорость: {stats.rows_per_second:,.0f} строк/сек")
        print(f"  - NaN значений: R={stats.nan_values['R']:,}, S={stats.nan_values['S']:,}, T={stats.nan_values['T']:,}")

        return True

    except Exception as e:
        logger.error(f"Ошибка загрузки файла {file_path}: {e}")
        print(f"✗ Ошибка загрузки {file_path.name}: {e}")
        return False


async def load_directory(
    directory_path: Path,
    pattern: str = "*.csv",
    equipment_id: str = None,
    sample_rate: int = 25600
):
    """Загрузить все CSV файлы из директории"""
    try:
        equipment_uuid = UUID(equipment_id) if equipment_id else None

        logger.info(f"Начинаем загрузку файлов из директории: {directory_path}")
        results = await load_csv_files_from_directory(
            directory_path=directory_path,
            pattern=pattern,
            equipment_id=equipment_uuid,
            sample_rate=sample_rate
        )

        print(f"\n📁 Обработка директории {directory_path} завершена:")
        print(f"  - Найдено файлов: {len(results)}")

        successful = 0
        failed = 0
        total_rows = 0

        for filename, stats in results.items():
            if stats:
                successful += 1
                total_rows += stats.processed_rows
                print(f"  ✓ {filename}: {stats.processed_rows:,} строк")
            else:
                failed += 1
                print(f"  ✗ {filename}: ОШИБКА")

        print(f"\nИтого:")
        print(f"  - Успешно: {successful} файлов")
        print(f"  - Ошибки: {failed} файлов")
        print(f"  - Всего строк: {total_rows:,}")

        return failed == 0

    except Exception as e:
        logger.error(f"Ошибка загрузки директории {directory_path}: {e}")
        print(f"✗ Ошибка загрузки директории: {e}")
        return False


def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Загрузка CSV файлов с токовыми сигналами в DiagMod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Загрузить один файл
  python scripts/load_csv_data.py data/raw/motor_001.csv

  # Загрузить с указанием оборудования
  python scripts/load_csv_data.py data/raw/motor_001.csv --equipment-id "550e8400-e29b-41d4-a716-446655440000"

  # Загрузить все файлы из директории
  python scripts/load_csv_data.py data/raw/ --pattern "motor_*.csv"

  # Настроить параметры обработки
  python scripts/load_csv_data.py data/raw/motor_001.csv --sample-rate 50000 --batch-size 5000

  # Посмотреть статистику без загрузки
  python scripts/load_csv_data.py data/raw/motor_001.csv --dry-run
        """)

    parser.add_argument(
        "path",
        type=str,
        help="Путь к CSV файлу или директории"
    )

    parser.add_argument(
        "--equipment-id",
        type=str,
        help="UUID оборудования (если не указан, определяется автоматически по имени файла)"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=25600,
        help="Частота дискретизации в Гц (по умолчанию: 25600)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Размер пачки для загрузки в БД (по умолчанию: 10000)"
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Паттерн имен файлов для директории (по умолчанию: *.csv)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать только статистику без загрузки в БД"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод"
    )

    args = parser.parse_args()

    # Настраиваем уровень логирования
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Проверяем путь
    path = Path(args.path)
    if not path.exists():
        print(f"✗ Путь не найден: {path}")
        sys.exit(1)

    if args.dry_run:
        print("🔍 Режим просмотра (данные не будут загружены в БД)")
        # TODO: Добавить функцию предварительного анализа
        print("Функция предварительного анализа будет добавлена позже")
        return

    # Запускаем загрузку
    async def run_loader():
        success = False

        if path.is_file():
            success = await load_single_file(
                file_path=path,
                equipment_id=args.equipment_id,
                sample_rate=args.sample_rate,
                batch_size=args.batch_size
            )
        elif path.is_dir():
            success = await load_directory(
                directory_path=path,
                pattern=args.pattern,
                equipment_id=args.equipment_id,
                sample_rate=args.sample_rate
            )
        else:
            print(f"✗ Неподдерживаемый тип пути: {path}")
            return False

        return success

    try:
        success = asyncio.run(run_loader())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠ Загрузка прервана пользователем")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        print(f"✗ Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
