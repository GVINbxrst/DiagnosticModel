#!/usr/bin/env python3
"""
CLI интерфейс для управления Celery Worker

Этот скрипт позволяет управлять Celery worker из командной строки:
- Запуск/остановка worker
- Мониторинг состояния задач
- Запуск отдельных задач
- Проверка состояния системы
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

# Добавляем корневую папку в путь
sys.path.append(str(Path(__file__).parent.parent))

from celery import Celery
from celery.result import AsyncResult
from src.worker.config import celery_app, get_worker_info
from src.worker.tasks import (
    process_raw, detect_anomalies, forecast_trend,
    cleanup_old_data, retrain_models
)
from src.worker.specialized_tasks import (
    batch_process_directory, process_equipment_workflow,
    health_check_system, daily_equipment_report
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def start_worker(
    queues: List[str] = None,
    concurrency: int = 4,
    loglevel: str = 'info',
    max_tasks_per_child: int = 1000
):
    """
    Запуск Celery worker

    Args:
        queues: Список очередей для обработки
        concurrency: Количество параллельных процессов
        loglevel: Уровень логирования
        max_tasks_per_child: Максимум задач на процесс
    """
    if queues is None:
        queues = ['processing', 'ml', 'batch', 'maintenance', 'monitoring']

    print(f"🚀 Запуск Celery Worker...")
    print(f"📋 Очереди: {', '.join(queues)}")
    print(f"⚡ Concurrency: {concurrency}")
    print(f"📝 Log level: {loglevel}")

    # Запускаем worker
    celery_app.worker_main([
        'worker',
        f'--queues={",".join(queues)}',
        f'--concurrency={concurrency}',
        f'--loglevel={loglevel}',
        f'--max-tasks-per-child={max_tasks_per_child}',
        '--without-gossip',
        '--without-mingle',
        '--without-heartbeat'
    ])


def start_beat(loglevel: str = 'info'):
    """
    Запуск Celery Beat scheduler

    Args:
        loglevel: Уровень логирования
    """
    print(f"⏰ Запуск Celery Beat scheduler...")
    print(f"📝 Log level: {loglevel}")

    # Запускаем beat
    celery_app.start([
        'beat',
        f'--loglevel={loglevel}',
        '--pidfile=',
        '--schedule=/tmp/celerybeat-schedule'
    ])


def get_worker_status() -> Dict:
    """Получение статуса worker"""

    try:
        # Проверяем активные worker'ы
        inspect = celery_app.control.inspect()

        active_nodes = inspect.active()
        if not active_nodes:
            return {
                'status': 'offline',
                'workers': 0,
                'message': 'Нет активных worker процессов'
            }

        # Получаем статистику
        stats = inspect.stats()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        worker_count = len(active_nodes.keys())
        total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        total_scheduled_tasks = sum(len(tasks) for tasks in scheduled_tasks.values()) if scheduled_tasks else 0

        return {
            'status': 'online',
            'workers': worker_count,
            'active_tasks': total_active_tasks,
            'scheduled_tasks': total_scheduled_tasks,
            'worker_nodes': list(active_nodes.keys()),
            'statistics': stats
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Ошибка получения статуса: {e}'
        }


def submit_task(task_name: str, args: List = None, kwargs: Dict = None) -> str:
    """
    Отправка задачи на выполнение

    Args:
        task_name: Имя задачи
        args: Позиционные аргументы
        kwargs: Именованные аргументы

    Returns:
        ID задачи
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    # Мапинг задач
    task_map = {
        'process_raw': process_raw,
        'detect_anomalies': detect_anomalies,
        'forecast_trend': forecast_trend,
        'cleanup_old_data': cleanup_old_data,
        'retrain_models': retrain_models,
        'batch_process_directory': batch_process_directory,
        'process_equipment_workflow': process_equipment_workflow,
        'health_check_system': health_check_system,
        'daily_equipment_report': daily_equipment_report
    }

    if task_name not in task_map:
        raise ValueError(f"Неизвестная задача: {task_name}")

    task = task_map[task_name]
    result = task.delay(*args, **kwargs)

    print(f"✅ Задача {task_name} отправлена с ID: {result.id}")
    return result.id


def get_task_status(task_id: str) -> Dict:
    """
    Получение статуса задачи

    Args:
        task_id: ID задачи

    Returns:
        Статус задачи
    """
    result = AsyncResult(task_id, app=celery_app)

    return {
        'task_id': task_id,
        'status': result.status,
        'result': result.result,
        'traceback': result.traceback,
        'successful': result.successful(),
        'failed': result.failed()
    }


def list_tasks(limit: int = 10) -> List[Dict]:
    """
    Список активных задач

    Args:
        limit: Максимальное количество задач

    Returns:
        Список задач
    """
    inspect = celery_app.control.inspect()

    try:
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        all_tasks = []

        # Активные задачи
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    all_tasks.append({
                        'id': task['id'],
                        'name': task['name'],
                        'worker': worker,
                        'status': 'active',
                        'args': task.get('args', []),
                        'kwargs': task.get('kwargs', {})
                    })

        # Запланированные задачи
        if scheduled_tasks:
            for worker, tasks in scheduled_tasks.items():
                for task in tasks:
                    all_tasks.append({
                        'id': task['request']['id'],
                        'name': task['request']['task'],
                        'worker': worker,
                        'status': 'scheduled',
                        'eta': task.get('eta'),
                        'args': task['request'].get('args', []),
                        'kwargs': task['request'].get('kwargs', {})
                    })

        return all_tasks[:limit]

    except Exception as e:
        logger.error(f"Ошибка получения списка задач: {e}")
        return []


def purge_tasks(queue_name: str = None):
    """
    Очистка очереди задач

    Args:
        queue_name: Имя очереди (None для всех)
    """
    if queue_name:
        result = celery_app.control.purge()
        print(f"🗑️  Очистка очереди {queue_name}: {result}")
    else:
        result = celery_app.control.purge()
        print(f"🗑️  Очистка всех очередей: {result}")


def show_system_health():
    """Показать состояние системы"""

    print("🏥 Проверка состояния системы...")

    # Отправляем задачу проверки
    result = health_check_system.delay()

    print(f"⏳ Ожидание результата задачи {result.id}...")

    # Ждем результат до 30 секунд
    try:
        health_data = result.get(timeout=30)

        print("\n" + "="*60)
        print("📊 СОСТОЯНИЕ СИСТЕМЫ")
        print("="*60)

        stats = health_data.get('statistics', {})

        # Сырые сигналы
        raw_signals = stats.get('raw_signals', {})
        print(f"📡 Сырые сигналы: {raw_signals.get('total', 0)}")
        for status, count in raw_signals.get('by_status', {}).items():
            print(f"   {status}: {count}")

        # Признаки
        features = stats.get('features', {})
        print(f"🔍 Извлеченные признаки: {features.get('total', 0)}")

        # Прогнозы за 24 часа
        predictions = stats.get('predictions_24h', {})
        print(f"📈 Прогнозы (24ч): {predictions.get('total', 0)}")
        print(f"🚨 Аномалии (24ч): {predictions.get('anomalies', 0)}")
        print(f"📊 Процент аномалий: {predictions.get('anomaly_rate', 0)*100:.1f}%")

        # Оборудование
        equipment = stats.get('equipment', {})
        print(f"⚙️  Активное оборудование: {equipment.get('active', 0)}")
        print(f"⚠️  С аномалиями: {equipment.get('with_recent_anomalies', 0)}")

        # Алерты
        alerts = health_data.get('alerts', [])
        if alerts:
            print(f"\n🚨 АЛЕРТЫ ({len(alerts)}):")
            for alert in alerts:
                level_emoji = {'warning': '🟡', 'critical': '🔴'}.get(alert['level'], '🔵')
                print(f"   {level_emoji} {alert['message']}")
        else:
            print(f"\n✅ Алертов нет")

        print("="*60)

    except Exception as e:
        print(f"❌ Ошибка получения состояния системы: {e}")


def main():
    """Главная функция CLI"""

    parser = argparse.ArgumentParser(
        description="Управление Celery Worker для диагностики двигателей",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python scripts/worker_cli.py worker --queues processing,ml --concurrency 4
  python scripts/worker_cli.py beat
  python scripts/worker_cli.py status
  python scripts/worker_cli.py submit process_raw --args '["signal-uuid"]'
  python scripts/worker_cli.py submit forecast_trend --args '["equipment-uuid"]'
  python scripts/worker_cli.py health
  python scripts/worker_cli.py tasks --limit 20
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    # Worker команда
    worker_parser = subparsers.add_parser('worker', help='Запуск worker процесса')
    worker_parser.add_argument(
        '--queues',
        default='processing,ml,batch,maintenance,monitoring',
        help='Список очередей через запятую'
    )
    worker_parser.add_argument(
        '--concurrency',
        type=int,
        default=4,
        help='Количество параллельных процессов'
    )
    worker_parser.add_argument(
        '--loglevel',
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Уровень логирования'
    )
    worker_parser.add_argument(
        '--max-tasks-per-child',
        type=int,
        default=1000,
        help='Максимум задач на процесс'
    )

    # Beat команда
    beat_parser = subparsers.add_parser('beat', help='Запуск beat scheduler')
    beat_parser.add_argument(
        '--loglevel',
        default='info',
        choices=['debug', 'info', 'warning', 'error'],
        help='Уровень логирования'
    )

    # Status команда
    subparsers.add_parser('status', help='Статус worker')

    # Submit команда
    submit_parser = subparsers.add_parser('submit', help='Отправить задачу')
    submit_parser.add_argument('task_name', help='Имя задачи')
    submit_parser.add_argument('--args', help='Аргументы в JSON формате')
    submit_parser.add_argument('--kwargs', help='Именованные аргументы в JSON формате')

    # Task status команда
    task_status_parser = subparsers.add_parser('task-status', help='Статус задачи')
    task_status_parser.add_argument('task_id', help='ID задачи')

    # Tasks список команда
    tasks_parser = subparsers.add_parser('tasks', help='Список активных задач')
    tasks_parser.add_argument('--limit', type=int, default=10, help='Максимум задач')

    # Purge команда
    purge_parser = subparsers.add_parser('purge', help='Очистить очереди')
    purge_parser.add_argument('--queue', help='Имя очереди (все если не указано)')

    # Health команда
    subparsers.add_parser('health', help='Состояние системы')

    # Info команда
    subparsers.add_parser('info', help='Информация о конфигурации')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'worker':
            queues = args.queues.split(',') if args.queues else None
            start_worker(
                queues=queues,
                concurrency=args.concurrency,
                loglevel=args.loglevel,
                max_tasks_per_child=args.max_tasks_per_child
            )

        elif args.command == 'beat':
            start_beat(loglevel=args.loglevel)

        elif args.command == 'status':
            status = get_worker_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))

        elif args.command == 'submit':
            task_args = json.loads(args.args) if args.args else []
            task_kwargs = json.loads(args.kwargs) if args.kwargs else {}
            task_id = submit_task(args.task_name, task_args, task_kwargs)
            print(f"Task ID: {task_id}")

        elif args.command == 'task-status':
            status = get_task_status(args.task_id)
            print(json.dumps(status, indent=2, ensure_ascii=False, default=str))

        elif args.command == 'tasks':
            tasks = list_tasks(args.limit)
            print(json.dumps(tasks, indent=2, ensure_ascii=False, default=str))

        elif args.command == 'purge':
            purge_tasks(args.queue)

        elif args.command == 'health':
            show_system_health()

        elif args.command == 'info':
            info = get_worker_info()
            print(json.dumps(info, indent=2, ensure_ascii=False))

    except KeyboardInterrupt:
        print("\n⏹️  Прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
