"""Точка входа Celery: экспортирует celery_app из config и задачи.

Периодические задачи определяются в src.worker.config (консолидация)."""

from src.worker.config import celery_app, get_worker_info  # noqa: F401

# Задачи не импортируем напрямую, чтобы избежать цикла с пакетом tasks.
# Пользователи должны импортировать их из src.worker.tasks

__all__ = ['celery_app', 'get_worker_info']
