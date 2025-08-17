#!/usr/bin/env python3
"""CLI обучения актуальных моделей аномалий (DBSCAN + PCA + streaming baseline)."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

from src.ml.train import train_anomaly_models, DEFAULT_CONTAMINATION
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def train_models_cli(
    equipment_ids: Optional[List[str]] = None,
    contamination: float = DEFAULT_CONTAMINATION,
    output_dir: Optional[str] = None,
    visualizations: bool = True
):
    """Обучить модели аномалий через CLI"""
    try:
        logger.info("Запуск обучения моделей аномалий")

        if equipment_ids:
            logger.info(f"Фильтрация по оборудованию: {equipment_ids}")

        results = await train_anomaly_models(
            equipment_ids=equipment_ids,
            contamination=contamination,
            output_dir=output_dir
        )

        # Выводим детальные результаты
        print(f"\n🎯 Обучение моделей завершено успешно!")
        print(f"📊 Статистика обучения:")
        print(f"  └─ Версия модели: {results['model_version']}")

    # Legacy IsolationForest удалён – блок пропущен

        # Результаты DBSCAN
        db_results = results['dbscan']
        print(f"\n🔍 DBSCAN:")
        print(f"  ├─ Кластеров: {db_results['n_clusters']}")
        print(f"  ├─ Аномалий: {db_results['n_anomalies']}")
        print(f"  └─ Доля аномалий: {db_results['anomaly_ratio']:.2%}")

        # Результаты PCA
        pca_results = results['pca']
        print(f"\n📈 PCA (снижение размерности):")
        print(f"  ├─ PC1: {pca_results['explained_variance_ratio'][0]:.1%} дисперсии")
        print(f"  ├─ PC2: {pca_results['explained_variance_ratio'][1]:.1%} дисперсии")
        print(f"  └─ Общая объясненная дисперсия: {pca_results['total_variance_explained']:.1%}")

        # Важность признаков
    # PCA вклад выводится через визуализации / метаданные

        # Визуализации
        if 'visualizations' in results:
            viz_paths = results['visualizations']
            print(f"\n📊 Созданные визуализации:")
            for viz_name, viz_path in viz_paths.items():
                print(f"  ├─ {viz_name}: {viz_path}")

        # Расположение моделей
        models_base_path = Path(output_dir) if output_dir else Path("./models/anomaly_detection")
        print(f"\n💾 Модели сохранены в:")
        print(f"  ├─ Версия: {models_base_path}/{results['model_version']}")
        print(f"  └─ Последняя: {models_base_path}/latest")

        return True

    except Exception as e:
        logger.error(f"Ошибка обучения моделей: {e}")
        print(f"\n❌ Ошибка обучения: {e}")
        return False


def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Обучение моделей аномалий для диагностики двигателей DiagMod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обучить модели на всех данных
  python scripts/train_models.py

  # Обучить только на данных конкретного оборудования
  python scripts/train_models.py --equipment-ids EQ_2025_000001 EQ_2025_000002

  # Настроить долю ожидаемых аномалий
  python scripts/train_models.py --contamination 0.05

  # Сохранить в кастомную директорию
  python scripts/train_models.py --output-dir ./custom_models

  # Обучить без создания визуализаций (быстрее)
  python scripts/train_models.py --no-visualizations
        """)

    parser.add_argument(
        "--equipment-ids",
        nargs="+",
        help="Список ID оборудования для обучения (например: EQ_2025_000001 EQ_2025_000002)"
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
        help=f"Ожидаемая доля аномалий (0.0-0.5, по умолчанию: {DEFAULT_CONTAMINATION})"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Директория для сохранения моделей (по умолчанию: ./models/anomaly_detection)"
    )

    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Не создавать визуализации (ускоряет обучение)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод"
    )

    args = parser.parse_args()

    # Валидация параметров
    if args.contamination < 0.0 or args.contamination > 0.5:
        print("❌ Доля аномалий должна быть в диапазоне [0.0, 0.5]")
        sys.exit(1)

    if args.output_dir and not Path(args.output_dir).parent.exists():
        print(f"❌ Родительская директория {args.output_dir} не существует")
        sys.exit(1)

    # Настраиваем логирование
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Запускаем обучение
    async def run_training():
        return await train_models_cli(
            equipment_ids=args.equipment_ids,
            contamination=args.contamination,
            output_dir=args.output_dir,
            visualizations=not args.no_visualizations
        )

    try:
        print("🚀 Запускаем обучение моделей аномалий...")
        success = asyncio.run(run_training())

        if success:
            print("\n✅ Обучение завершено успешно!")
            print("💡 Модели готовы для использования в системе диагностики")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Обучение прервано пользователем")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        print(f"\n❌ Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
