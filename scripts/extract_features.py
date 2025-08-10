#!/usr/bin/env python3
"""
Скрипт для извлечения признаков из токовых сигналов DiagMod
Обрабатывает сырые сигналы и извлекает статистические и частотные характеристики
"""

import argparse
import asyncio
import sys
from uuid import UUID

from src.data_processing.feature_extraction import (
    FeatureExtractor,
    process_unprocessed_signals,
    DEFAULT_SAMPLE_RATE
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def extract_features_from_signal(
    signal_id: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_ms: int = 1000,
    overlap: float = 0.5
):
    """Извлечь признаки из конкретного сигнала"""
    try:
        signal_uuid = UUID(signal_id)

        extractor = FeatureExtractor(sample_rate=sample_rate)

        logger.info(f"Начинаем обработку сигнала: {signal_id}")
        feature_ids = await extractor.process_raw_signal(
            signal_uuid,
            window_duration_ms=window_ms,
            overlap_ratio=overlap
        )

        print(f"✓ Сигнал {signal_id} обработан успешно:")
        print(f"  - Создано записей признаков: {len(feature_ids)}")
        print(f"  - Параметры: окно {window_ms}мс, перекрытие {overlap:.0%}")

        return True

    except Exception as e:
        logger.error(f"Ошибка обработки сигнала {signal_id}: {e}")
        print(f"✗ Ошибка обработки сигнала: {e}")
        return False


async def batch_process_signals(
    limit: int = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    window_ms: int = 1000,
    overlap: float = 0.5
):
    """Массовая обработка необработанных сигналов"""
    try:
        logger.info("Начинаем массовую обработку сигналов")

        stats = await process_unprocessed_signals(
            limit=limit,
            window_duration_ms=window_ms,
            overlap_ratio=overlap
        )

        print(f"✓ Массовая обработка завершена:")
        print(f"  - Обработано сигналов: {stats['processed_signals']}")
        print(f"  - Создано записей признаков: {stats['created_features']}")
        print(f"  - Ошибок: {stats['errors']}")

        if stats['created_features'] > 0:
            avg_features = stats['created_features'] / max(stats['processed_signals'], 1)
            print(f"  - Среднее признаков на сигнал: {avg_features:.1f}")

        return stats['errors'] == 0

    except Exception as e:
        logger.error(f"Ошибка массовой обработки: {e}")
        print(f"✗ Ошибка массовой обработки: {e}")
        return False


def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Извлечение признаков из токовых сигналов DiagMod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обработать все необработанные сигналы
  python scripts/extract_features.py

  # Обработать максимум 10 сигналов
  python scripts/extract_features.py --limit 10

  # Обработать конкретный сигнал
  python scripts/extract_features.py --signal-id "550e8400-e29b-41d4-a716-446655440000"

  # Настроить параметры окна
  python scripts/extract_features.py --window-ms 2000 --overlap 0.75

  # Указать частоту дискретизации
  python scripts/extract_features.py --sample-rate 50000
        """)

    parser.add_argument(
        "--signal-id",
        type=str,
        help="UUID конкретного сигнала для обработки"
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Максимальное количество сигналов для массовой обработки"
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Частота дискретизации в Гц (по умолчанию: {DEFAULT_SAMPLE_RATE})"
    )

    parser.add_argument(
        "--window-ms",
        type=int,
        default=1000,
        help="Длительность окна анализа в миллисекундах (по умолчанию: 1000)"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Коэффициент перекрытия окон 0.0-1.0 (по умолчанию: 0.5)"
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

    # Валидация параметров
    if args.overlap < 0.0 or args.overlap >= 1.0:
        print("✗ Коэффициент перекрытия должен быть в диапазоне [0.0, 1.0)")
        sys.exit(1)

    if args.window_ms <= 0:
        print("✗ Длительность окна должна быть положительной")
        sys.exit(1)

    if args.sample_rate <= 0:
        print("✗ Частота дискретизации должна быть положительной")
        sys.exit(1)

    # Запускаем обработку
    async def run_extraction():
        success = False

        if args.signal_id:
            # Обрабатываем конкретный сигнал
            success = await extract_features_from_signal(
                signal_id=args.signal_id,
                sample_rate=args.sample_rate,
                window_ms=args.window_ms,
                overlap=args.overlap
            )
        else:
            # Массовая обработка
            success = await batch_process_signals(
                limit=args.limit,
                sample_rate=args.sample_rate,
                window_ms=args.window_ms,
                overlap=args.overlap
            )

        return success

    try:
        success = asyncio.run(run_extraction())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠ Обработка прервана пользователем")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        print(f"✗ Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
