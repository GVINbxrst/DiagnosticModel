"""
Инициализационный модуль для задач Celery
"""

from src.worker.tasks.specialized_tasks import (
    batch_process_directory,
    process_equipment_workflow,
    health_check_system,
    daily_equipment_report
)

__all__ = [
    'batch_process_directory',
    'process_equipment_workflow',
    'health_check_system',
    'daily_equipment_report'
]
