#!/usr/bin/env python3
"""
Скрипт для запуска прогнозирования RMS трендов

Этот скрипт позволяет запускать прогнозирование из командной строки
для анализа трендов токовых сигналов двигателей.

Использование:
    python scripts/forecast_rms.py --equipment-id UUID --steps 24 --phases a,b,c
    python scripts/forecast_rms.py --all-equipment --steps 12
    python scripts/forecast_rms.py --equipment-id UUID --output forecast_results.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Optional
from uuid import UUID

# Добавляем корневую папку в путь
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.forecasting import (
    forecast_rms_trends,
    get_anomaly_probability,
    RMSTrendForecaster,
    DEFAULT_FORECAST_STEPS
)
from src.database.connection import get_async_session
from src.database.models import Equipment
from src.utils.logger import get_logger
from sqlalchemy import select

logger = get_logger(__name__)


async def get_all_equipment_ids() -> List[UUID]:
    """Получить список всех ID оборудования"""
    async with get_async_session() as session:
        query = select(Equipment.id).where(Equipment.is_active == True)
        result = await session.execute(query)
        return [row[0] for row in result.fetchall()]


async def forecast_single_equipment(
    equipment_id: UUID,
    forecast_steps: int,
    phases: List[str],
    output_file: Optional[str] = None
) -> dict:
    """Прогнозирование для одного оборудования"""
    logger.info(f"Начинаем прогнозирование для оборудования {equipment_id}")

    try:
        results = await forecast_rms_trends(
            equipment_id=equipment_id,
            forecast_steps=forecast_steps,
            phases=phases
        )

        # Сохраняем результаты в файл, если указан
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Результаты сохранены в {output_file}")

        return results

    except Exception as e:
        logger.error(f"Ошибка прогнозирования для {equipment_id}: {e}")
        return {'error': str(e), 'equipment_id': str(equipment_id)}


async def forecast_all_equipment(
    forecast_steps: int,
    phases: List[str],
    output_dir: str = "data/forecasts"
) -> dict:
    """Прогнозирование для всего оборудования"""
    logger.info("Начинаем массовое прогнозирование")

    # Создаем папку для результатов
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    equipment_ids = await get_all_equipment_ids()
    logger.info(f"Найдено {len(equipment_ids)} единиц оборудования")

    all_results = {
        'timestamp': datetime.now(UTC).isoformat(),
        'total_equipment': len(equipment_ids),
        'forecast_steps': forecast_steps,
        'phases': phases,
        'results': {},
        'summary': {
            'successful': 0,
            'failed': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0
        }
    }

    for equipment_id in equipment_ids:
        try:
            results = await forecast_single_equipment(
                equipment_id=equipment_id,
                forecast_steps=forecast_steps,
                phases=phases
            )

            # Сохраняем индивидуальный результат
            individual_file = Path(output_dir) / f"forecast_{equipment_id}.json"
            with open(individual_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

            if 'error' in results:
                all_results['summary']['failed'] += 1
            else:
                all_results['summary']['successful'] += 1

                # Классификация риска
                max_prob = results.get('summary', {}).get('max_anomaly_probability', 0)
                if max_prob >= 0.6:
                    all_results['summary']['high_risk'] += 1
                elif max_prob >= 0.3:
                    all_results['summary']['medium_risk'] += 1
                else:
                    all_results['summary']['low_risk'] += 1

            all_results['results'][str(equipment_id)] = {
                'status': 'success' if 'error' not in results else 'failed',
                'max_anomaly_probability': results.get('summary', {}).get('max_anomaly_probability'),
                'recommendation': results.get('summary', {}).get('recommendation'),
                'file': str(individual_file)
            }

            logger.info(f"Обработано оборудование {equipment_id}")

        except Exception as e:
            logger.error(f"Критическая ошибка для {equipment_id}: {e}")
            all_results['summary']['failed'] += 1
            all_results['results'][str(equipment_id)] = {
                'status': 'critical_error',
                'error': str(e)
            }

    # Сохраняем сводный отчет
    summary_file = Path(output_dir) / f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Массовое прогнозирование завершено. Сводка сохранена в {summary_file}")

    return all_results


async def quick_anomaly_check(equipment_id: UUID, time_windows: int = 24) -> float:
    """Быстрая проверка вероятности аномалии"""
    logger.info(f"Быстрая проверка аномалии для {equipment_id}")

    try:
        probability = await get_anomaly_probability(equipment_id, time_windows)

        status = "НОРМАЛЬНО"
        if probability >= 0.8:
            status = "КРИТИЧНО"
        elif probability >= 0.6:
            status = "ВЫСОКИЙ РИСК"
        elif probability >= 0.3:
            status = "СРЕДНИЙ РИСК"
        elif probability >= 0.1:
            status = "НИЗКИЙ РИСК"

        print(f"Оборудование {equipment_id}:")
        print(f"  Вероятность аномалии: {probability:.3f}")
        print(f"  Статус: {status}")

        return probability

    except Exception as e:
        logger.error(f"Ошибка быстрой проверки: {e}")
        return 0.0


def print_results_summary(results: dict):
    """Вывод краткой сводки результатов"""
    if 'error' in results:
        print(f"❌ Ошибка: {results['error']}")
        return

    summary = results.get('summary', {})
    print("\n" + "="*60)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ ПРОГНОЗИРОВАНИЯ")
    print("="*60)

    print(f"🔧 Оборудование: {results.get('equipment_id', 'N/A')}")
    print(f"📅 Время анализа: {results.get('timestamp', 'N/A')}")
    print(f"📈 Шагов прогноза: {results.get('forecast_steps', 'N/A')}")

    if summary:
        print(f"\n🎯 МАКСИМАЛЬНАЯ ВЕРОЯТНОСТЬ АНОМАЛИИ: {summary.get('max_anomaly_probability', 0):.3f}")
        print(f"💡 РЕКОМЕНДАЦИЯ: {summary.get('recommendation', 'Не определена')}")
        print(f"✅ Успешных фаз: {summary.get('successful_phases', 0)}")
        print(f"⚠️  Шагов высокого риска: {summary.get('high_risk_steps', 0)}")

    # Детали по фазам
    phases = results.get('phases', {})
    if phases:
        print(f"\n📋 ДЕТАЛИ ПО ФАЗАМ:")
        for phase, phase_data in phases.items():
            if 'error' in phase_data:
                print(f"  ❌ Фаза {phase.upper()}: {phase_data['error']}")
            else:
                stats = phase_data.get('statistics', {})
                print(f"  ✅ Фаза {phase.upper()}:")
                print(f"     Наблюдений: {phase_data.get('n_observations', 0)}")
                print(f"     RMS среднее: {stats.get('mean', 0):.4f}")
                print(f"     RMS стд. откл.: {stats.get('std', 0):.4f}")

    print("="*60)


async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Прогнозирование RMS трендов для диагностики двигателей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/forecast_rms.py --equipment-id "123e4567-e89b-12d3-a456-426614174000" --steps 24
  python scripts/forecast_rms.py --all-equipment --steps 12 --output-dir data/forecasts
  python scripts/forecast_rms.py --quick-check "123e4567-e89b-12d3-a456-426614174000"
  python scripts/forecast_rms.py --equipment-id "123e4567-e89b-12d3-a456-426614174000" --phases a,b --output results.json
        """
    )

    # Основные параметры
    parser.add_argument(
        '--equipment-id',
        type=str,
        help='UUID оборудования для анализа'
    )

    parser.add_argument(
        '--all-equipment',
        action='store_true',
        help='Прогнозирование для всего оборудования'
    )

    parser.add_argument(
        '--quick-check',
        type=str,
        help='Быстрая проверка вероятности аномалии для указанного оборудования'
    )

    # Параметры прогнозирования
    parser.add_argument(
        '--steps',
        type=int,
        default=DEFAULT_FORECAST_STEPS,
        help=f'Количество шагов прогноза (по умолчанию: {DEFAULT_FORECAST_STEPS})'
    )

    parser.add_argument(
        '--phases',
        type=str,
        default='a,b,c',
        help='Фазы для анализа через запятую (по умолчанию: a,b,c)'
    )

    parser.add_argument(
        '--time-windows',
        type=int,
        default=24,
        help='Количество временных окон для быстрой проверки (по умолчанию: 24)'
    )

    # Параметры вывода
    parser.add_argument(
        '--output',
        type=str,
        help='Файл для сохранения результатов (JSON)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/forecasts',
        help='Папка для сохранения результатов массового прогнозирования'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Тихий режим (только ошибки)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод'
    )

    args = parser.parse_args()

    # Настройка логирования
    if args.quiet:
        logger.setLevel('ERROR')
    elif args.verbose:
        logger.setLevel('DEBUG')

    # Парсинг фаз
    phases = [phase.strip() for phase in args.phases.split(',') if phase.strip()]

    try:
        if args.quick_check:
            # Быстрая проверка
            equipment_id = UUID(args.quick_check)
            await quick_anomaly_check(equipment_id, args.time_windows)

        elif args.all_equipment:
            # Массовое прогнозирование
            results = await forecast_all_equipment(
                forecast_steps=args.steps,
                phases=phases,
                output_dir=args.output_dir
            )

            print(f"\n✅ Массовое прогнозирование завершено!")
            print(f"📊 Успешно: {results['summary']['successful']}")
            print(f"❌ Ошибок: {results['summary']['failed']}")
            print(f"🔴 Высокий риск: {results['summary']['high_risk']}")
            print(f"🟡 Средний риск: {results['summary']['medium_risk']}")
            print(f"🟢 Низкий риск: {results['summary']['low_risk']}")

        elif args.equipment_id:
            # Прогнозирование для одного оборудования
            equipment_id = UUID(args.equipment_id)
            results = await forecast_single_equipment(
                equipment_id=equipment_id,
                forecast_steps=args.steps,
                phases=phases,
                output_file=args.output
            )

            if not args.quiet:
                print_results_summary(results)

        else:
            parser.print_help()
            return

    except ValueError as e:
        print(f"❌ Ошибка в параметрах: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
