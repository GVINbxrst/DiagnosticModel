# Мониторинг Celery Worker (метрики + логирование)
import time
import functools
from typing import Dict, Any, Callable
from celery import Celery
from celery.signals import (
    task_prerun, task_postrun, task_failure, task_success,
    worker_ready, worker_shutdown
)

from src.utils.metrics import (
    track_worker_task, worker_tasks_total, worker_task_duration_seconds,
    worker_active_tasks, increment_counter, observe_histogram, set_gauge
)
from src.utils.logger import get_logger, get_audit_logger

logger = get_logger(__name__)
audit_logger = get_audit_logger()


class WorkerMetricsCollector:
    # Коллектор метрик Worker

    def __init__(self):
        self.active_tasks_count = 0
        self.task_start_times = {}
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        # Настройка сигналов Celery

        @task_prerun.connect
        def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwargs_extra):
            # Начало задачи
            self.on_task_prerun(task_id, task.__name__, args, kwargs)

        @task_postrun.connect
        def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None,
                               retval=None, state=None, **kwargs_extra):
            # Завершение задачи
            self.on_task_postrun(task_id, task.__name__, state, retval)

        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
            # Ошибка задачи
            self.on_task_failure(task_id, sender.__name__, exception, traceback)

        @task_success.connect
        def task_success_handler(sender=None, result=None, **kwargs):
            # Успех задачи
            self.on_task_success(sender.__name__, result)

        @worker_ready.connect
        def worker_ready_handler(sender=None, **kwargs):
            # Готовность воркера
            self.on_worker_ready(sender)

        @worker_shutdown.connect
        def worker_shutdown_handler(sender=None, **kwargs):
            # Остановка воркера
            self.on_worker_shutdown(sender)

    def on_task_prerun(self, task_id: str, task_name: str, args: tuple, kwargs: dict):
        # Начало задачи
        self.active_tasks_count += 1
        self.task_start_times[task_id] = time.time()

        # Обновляем метрику активных задач
        set_gauge('worker_active_tasks', self.active_tasks_count)

        # Логируем начало задачи
        logger.info(
            f"🔄 Начало выполнения задачи {task_name}",
            extra={
                'event_type': 'task_start',
                'task_id': task_id,
                'task_name': task_name,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }
        )

    def on_task_postrun(self, task_id: str, task_name: str, state: str, result: Any):
        # Завершение задачи
        self.active_tasks_count = max(0, self.active_tasks_count - 1)

        # Вычисляем время выполнения
        start_time = self.task_start_times.pop(task_id, None)
        duration = time.time() - start_time if start_time else 0

        # Обновляем метрики
        set_gauge('worker_active_tasks', self.active_tasks_count)

        increment_counter(
            'worker_tasks_total',
            {'task_name': task_name, 'status': state or 'unknown'}
        )

        observe_histogram(
            'worker_task_duration_seconds',
            duration,
            {'task_name': task_name}
        )

        # Логируем завершение задачи
        logger.info(
            f"✅ Завершение задачи {task_name}",
            extra={
                'event_type': 'task_complete',
                'task_id': task_id,
                'task_name': task_name,
                'state': state,
                'duration_seconds': duration,
                'result_type': type(result).__name__ if result else 'None'
            }
        )

        # Логируем медленные задачи
        if duration > 60:  # Задачи дольше минуты
            logger.warning(
                f"🐌 Медленная задача {task_name} выполнялась {duration:.2f}s",
                extra={
                    'event_type': 'slow_task',
                    'task_id': task_id,
                    'task_name': task_name,
                    'duration_seconds': duration
                }
            )

    def on_task_failure(self, task_id: str, task_name: str, exception: Exception, traceback: str):
        # Ошибка задачи
        self.active_tasks_count = max(0, self.active_tasks_count - 1)

        # Вычисляем время выполнения до ошибки
        start_time = self.task_start_times.pop(task_id, None)
        duration = time.time() - start_time if start_time else 0

        # Обновляем метрики
        set_gauge('worker_active_tasks', self.active_tasks_count)

        increment_counter(
            'worker_tasks_total',
            {'task_name': task_name, 'status': 'failure'}
        )

        # Логируем ошибку
        logger.error(
            f"❌ Ошибка в задаче {task_name}: {exception}",
            extra={
                'event_type': 'task_error',
                'task_id': task_id,
                'task_name': task_name,
                'error_type': type(exception).__name__,
                'error_message': str(exception),
                'duration_seconds': duration
            },
            exc_info=True
        )

    def on_task_success(self, task_name: str, result: Any):
        # Успех задачи
        logger.debug(
            f"✅ Успешное выполнение задачи {task_name}",
            extra={
                'event_type': 'task_success',
                'task_name': task_name,
                'result_type': type(result).__name__ if result else 'None'
            }
        )

    def on_worker_ready(self, sender):
        # Воркер готов
        worker_hostname = getattr(sender, 'hostname', 'unknown')
        logger.info(
            f"🟢 Worker {worker_hostname} готов к работе",
            extra={
                'event_type': 'worker_ready',
                'worker_hostname': worker_hostname
            }
        )

    def on_worker_shutdown(self, sender):
        # Остановка воркера
        worker_hostname = getattr(sender, 'hostname', 'unknown')
        logger.info(
            f"🔴 Worker {worker_hostname} завершает работу",
            extra={
                'event_type': 'worker_shutdown',
                'worker_hostname': worker_hostname
            }
        )


def track_task_metrics(task_name: str = None):
    # Декоратор отслеживания метрик задач
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            task_name_resolved = task_name or func.__name__
            status = 'success'

            try:
                # Логируем начало выполнения
                logger.info(
                    f"🔄 Выполнение задачи {task_name_resolved}",
                    extra={
                        'event_type': 'task_execution',
                        'task_name': task_name_resolved,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                )

                result = func(*args, **kwargs)
                return result

            except Exception as e:
                status = 'error'
                logger.error(
                    f"❌ Ошибка в задаче {task_name_resolved}: {e}",
                    extra={
                        'event_type': 'task_error',
                        'task_name': task_name_resolved,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                raise

            finally:
                duration = time.time() - start_time

                # Обновляем метрики
                increment_counter(
                    'worker_tasks_total',
                    {'task_name': task_name_resolved, 'status': status}
                )

                observe_histogram(
                    'worker_task_duration_seconds',
                    duration,
                    {'task_name': task_name_resolved}
                )

                logger.info(
                    f"✅ Задача {task_name_resolved} завершена за {duration:.2f}s",
                    extra={
                        'event_type': 'task_completed',
                        'task_name': task_name_resolved,
                        'duration_seconds': duration,
                        'status': status
                    }
                )

        return wrapper
    return decorator


def setup_worker_monitoring(celery_app: Celery):
    # Настройка мониторинга Celery
    # Создаем коллектор метрик
    metrics_collector = WorkerMetricsCollector()

    # Добавляем middleware для всех задач
    @celery_app.task(bind=True)
    def monitored_task(self, original_task, *args, **kwargs):
        """Обертка для мониторинга задач"""
        with track_task_metrics(original_task.__name__):
            return original_task(*args, **kwargs)

    logger.info("📊 Мониторинг Worker настроен")
    return metrics_collector


# Создаем HTTP сервер для метрик Worker
def create_worker_metrics_server(port: int = 8002):
    # HTTP сервер метрик Worker
    from prometheus_client import start_http_server, generate_latest
    from src.utils.metrics import get_all_metrics

    try:
        start_http_server(port)
        logger.info(f"📊 HTTP сервер метрик Worker запущен на порту {port}")
        logger.info(f"🔗 Метрики доступны по адресу: http://localhost:{port}/metrics")
    except Exception as e:
        logger.error(f"❌ Ошибка запуска HTTP сервера метрик: {e}")


## Удалён Flask endpoint (избыточная зависимость). Используйте create_worker_metrics_server/start_http_server.


# Глобальный экземпляр коллектора
worker_metrics_collector = None

def get_worker_metrics_collector() -> WorkerMetricsCollector:
    # Получить глобальный коллектор
    global worker_metrics_collector
    if worker_metrics_collector is None:
        worker_metrics_collector = WorkerMetricsCollector()
    return worker_metrics_collector
