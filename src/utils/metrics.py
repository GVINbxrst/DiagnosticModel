"""
Система Prometheus метрик для DiagMod
Сбор и экспорт метрик для мониторинга производительности и состояния системы
"""

import time
from functools import wraps
from typing import Dict, Optional, Callable, Any
import asyncio
from datetime import datetime

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest,
    multiprocess, CONTENT_TYPE_LATEST
)
import psutil
import logging

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()
# --- Минимальные публичные API ---
def increment_counter(name: str, labels: Optional[Dict[str, str]] = None):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для increment_counter")
        return
    if labels:
        metric.labels(**labels).inc()
    else:
        metric.inc()

def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для observe_histogram")
        return
    if labels:
        metric.labels(**labels).observe(value)
    else:
        metric.observe(value)

def get_metrics() -> bytes:
    return generate_latest(REGISTRY)

def observe_latency(metric_name: str, labels: Optional[Dict[str, str]] = None):
    def decorator(func: Callable):
        is_coroutine = asyncio.iscoroutinefunction(func)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metric = globals().get(metric_name)
                if metric:
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                metric = globals().get(metric_name)
                if metric:
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
        return async_wrapper if is_coroutine else sync_wrapper
    return decorator

# Основные метрики для API
api_requests_total = Counter(
    'api_requests_total',
    'Общее количество API запросов',
    ['method', 'endpoint', 'status_code', 'user_role'],
    registry=REGISTRY
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'Время выполнения API запросов в секундах',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Метрики для обнаружения аномалий
anomalies_detected_total = Counter(
    'anomalies_detected_total',
    'Общее количество обнаруженных аномалий',
    ['equipment_id', 'model_name', 'defect_type'],
    registry=REGISTRY
)

anomaly_detection_duration_seconds = Histogram(
    'anomaly_detection_duration_seconds',
    'Время выполнения обнаружения аномалий в секундах',
    ['model_name', 'equipment_id'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=REGISTRY
)

# Метрики для прогнозирования
forecast_latency_seconds = Histogram(
    'forecast_latency_seconds',
    'Время генерации прогнозов в секундах',
    ['model_name', 'equipment_id', 'forecast_horizon'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    registry=REGISTRY
)

forecasts_generated_total = Counter(
    'forecasts_generated_total',
    'Общее количество сгенерированных прогнозов',
    ['model_name', 'equipment_id', 'status'],
    registry=REGISTRY
)

# Метрики для обработки данных
csv_files_processed_total = Counter(
    'csv_files_processed_total',
    'Общее количество обработанных CSV файлов',
    ['equipment_id', 'status'],
    registry=REGISTRY
)

csv_processing_duration_seconds = Histogram(
    'csv_processing_duration_seconds',
    'Время обработки CSV файлов в секундах',
    ['equipment_id'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    registry=REGISTRY
)

data_points_processed_total = Counter(
    'data_points_processed_total',
    'Общее количество обработанных точек данных',
    ['equipment_id', 'phase'],
    registry=REGISTRY
)

# Метрики для Celery Worker
worker_tasks_total = Counter(
    'worker_tasks_total',
    'Общее количество задач воркера',
    ['task_name', 'status'],
    registry=REGISTRY
)

worker_task_duration_seconds = Histogram(
    'worker_task_duration_seconds',
    'Время выполнения задач воркера в секундах',
    ['task_name'],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
    registry=REGISTRY
)

worker_active_tasks = Gauge(
    'worker_active_tasks',
    'Количество активных задач воркера',
    registry=REGISTRY
)

# Системные метрики
system_cpu_usage_percent = Gauge(
    'system_cpu_usage_percent',
    'Использование CPU системы в процентах',
    registry=REGISTRY
)

system_memory_usage_bytes = Gauge(
    'system_memory_usage_bytes',
    'Использование памяти системы в байтах',
    ['type'],  # used, available, total
    registry=REGISTRY
)

system_disk_usage_bytes = Gauge(
    'system_disk_usage_bytes',
    'Использование диска системы в байтах',
    ['path', 'type'],  # used, free, total
    registry=REGISTRY
)

# Метрики базы данных
database_connections_active = Gauge(
    'database_connections_active',
    'Количество активных подключений к базе данных',
    registry=REGISTRY
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Время выполнения запросов к базе данных в секундах',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

# Информационные метрики
app_info = Info(
    'diagmod_app_info',
    'Информация о приложении DiagMod',
    registry=REGISTRY
)


class MetricsCollector:
    """Коллектор метрик для централизованного управления"""

    def __init__(self):
        self.start_time = time.time()
        self._update_app_info()
        self._start_system_metrics_collection()

    def _update_app_info(self):
        """Обновление информационных метрик приложения"""
        app_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'started_at': datetime.now().isoformat()
        })

    def _start_system_metrics_collection(self):
        """Начало сбора системных метрик"""
    # Эти метрики будут обновляться периодически (внешним планировщиком / background task)
    return None

    def update_system_metrics(self):
        """Обновление системных метрик"""
        try:
            # CPU метрики
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_usage_percent.set(cpu_percent)

    # --- Унифицированные утилиты ---

def increment(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Увеличить Counter по имени.
    Пример: increment('api_requests_total', {'method':'GET','endpoint':'/signals','status_code':'200','user_role':'engineer'})
    """
    metric = globals().get(metric_name)
    if metric is None:
        logger.warning(f"Метрика {metric_name} не найдена для increment")
        return
    if labels:
        metric.labels(**labels).inc()
    else:
        metric.inc()

# --- Public wrapper API (контракт) ---

def increment_counter(name: str, labels: Optional[Dict[str, str]] = None):
    """Alias для increment по контракту задачи."""
    increment(name, labels)


def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для observe_histogram")
        return
    try:
        if labels and getattr(metric, 'labels', None):
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    except Exception as e:  # noqa
        logger.debug(f"Не удалось observe {name}: {e}")


def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для set_gauge")
        return
    try:
        if labels and getattr(metric, 'labels', None):
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    except Exception as e:
        logger.debug(f"Не удалось set {name}: {e}")


def get_metrics() -> bytes:
    """Вернуть сериализованные метрики."""
    from prometheus_client import generate_latest  # локальный импорт
    return generate_latest(REGISTRY)


def get_all_metrics() -> bytes:  # совместимость с существующим импортом
    return get_metrics()


def observe_latency(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Декоратор измерения времени выполнения и записи в Histogram.

    Если переданы labels – используются они, иначе выполняется попытка авто-лейблинга.
    """
    def decorator(func: Callable):
        is_coroutine = asyncio.iscoroutinefunction(func) if 'asyncio' in globals() else hasattr(func, '__await__')

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                hist = globals().get(metric_name)
                if hist is not None:
                    try:
                        if labels:
                            hist.labels(**labels).observe(duration) if getattr(hist, '_labelnames', None) else hist.observe(duration)
                        else:
                            endpoint = kwargs.get('endpoint') or getattr(func, '__name__', 'unknown')
                            if hasattr(hist, 'labels') and hist._labelnames:  # type: ignore
                                labelnames = list(hist._labelnames)  # type: ignore
                                values = {}
                                for ln in labelnames:
                                    if ln == 'endpoint':
                                        values[ln] = endpoint
                                    elif ln == 'method':
                                        values[ln] = kwargs.get('method', 'AUTO')
                                    else:
                                        values[ln] = 'auto'
                                hist.labels(**values).observe(duration)
                            else:
                                hist.observe(duration)
                    except Exception as e:  # noqa
                        logger.debug(f"Не удалось записать latency в {metric_name}: {e}")

        async def async_wrapper(*args, **kwargs):  # type: ignore
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                hist = globals().get(metric_name)
                if hist is not None:
                    try:
                        if labels:
                            hist.labels(**labels).observe(duration) if getattr(hist, '_labelnames', None) else hist.observe(duration)
                        else:
                            endpoint = kwargs.get('endpoint') or getattr(func, '__name__', 'unknown')
                            if hasattr(hist, 'labels') and getattr(hist, '_labelnames', None):  # type: ignore
                                labelnames = list(hist._labelnames)  # type: ignore
                                values = {}
                                for ln in labelnames:
                                    if ln == 'endpoint':
                                        values[ln] = endpoint
                                    elif ln == 'method':
                                        values[ln] = kwargs.get('method', 'AUTO')
                                    else:
                                        values[ln] = 'auto'
                                hist.labels(**values).observe(duration)
                            else:
                                hist.observe(duration)
                    except Exception as e:  # noqa
                        logger.debug(f"Не удалось записать latency в {metric_name}: {e}")

        return async_wrapper if is_coroutine else sync_wrapper

    return decorator


__all__ = [
    'increment', 'increment_counter', 'observe_histogram', 'set_gauge', 'get_metrics', 'get_all_metrics', 'observe_latency'
]
            # Память
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.labels(type='used').set(memory.used)
            system_memory_usage_bytes.labels(type='available').set(memory.available)
            system_memory_usage_bytes.labels(type='total').set(memory.total)

            # Диск
            disk = psutil.disk_usage('/')
            system_disk_usage_bytes.labels(path='/', type='used').set(disk.used)
            system_disk_usage_bytes.labels(path='/', type='free').set(disk.free)
            system_disk_usage_bytes.labels(path='/', type='total').set(disk.total)

        except Exception as e:
            logger.error(f"Ошибка при обновлении системных метрик: {e}")

    def get_metrics(self) -> str:
        """Получение всех метрик в формате Prometheus"""
        self.update_system_metrics()
        return generate_latest(REGISTRY)


# Глобальный коллектор метрик
metrics_collector = MetricsCollector()


def track_api_request(method: str, endpoint: str, user_role: str = 'unknown'):
    """Декоратор для отслеживания API запросов"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status_code = 500  # По умолчанию ошибка

            try:
                result = await func(*args, **kwargs)
                status_code = getattr(result, 'status_code', 200)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time

                # Обновляем метрики
                api_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code),
                    user_role=user_role
                ).inc()

                api_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)

        return wrapper
    return decorator


def track_anomaly_detection(equipment_id: int, model_name: str):
    """Декоратор для отслеживания обнаружения аномалий"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Если результат содержит информацию об аномалии
                if isinstance(result, dict) and 'is_anomaly' in result:
                    if result['is_anomaly']:
                        defect_type = result.get('defect_type', 'unknown')
                        anomalies_detected_total.labels(
                            equipment_id=str(equipment_id),
                            model_name=model_name,
                            defect_type=defect_type
                        ).inc()

                return result
            finally:
                duration = time.time() - start_time
                anomaly_detection_duration_seconds.labels(
                    model_name=model_name,
                    equipment_id=str(equipment_id)
                ).observe(duration)

        return wrapper
    return decorator


def track_forecast_generation(equipment_id: int, model_name: str, forecast_horizon: int):
    """Декоратор для отслеживания генерации прогнозов"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time

                # Метрика времени выполнения
                forecast_latency_seconds.labels(
                    model_name=model_name,
                    equipment_id=str(equipment_id),
                    forecast_horizon=str(forecast_horizon)
                ).observe(duration)

                # Счетчик прогнозов
                forecasts_generated_total.labels(
                    model_name=model_name,
                    equipment_id=str(equipment_id),
                    status=status
                ).inc()

        return wrapper
    return decorator


def track_csv_processing(equipment_id: int):
    """Декоратор для отслеживания обработки CSV файлов"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            try:
                result = func(*args, **kwargs)

                # Подсчет обработанных точек данных
                if isinstance(result, dict) and 'processed_points' in result:
                    for phase, count in result['processed_points'].items():
                        data_points_processed_total.labels(
                            equipment_id=str(equipment_id),
                            phase=phase
                        ).inc(count)

                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time

                csv_files_processed_total.labels(
                    equipment_id=str(equipment_id),
                    status=status
                ).inc()

                csv_processing_duration_seconds.labels(
                    equipment_id=str(equipment_id)
                ).observe(duration)

        return wrapper
    return decorator


def track_worker_task(task_name: str):
    """Декоратор для отслеживания задач Celery"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'

            worker_active_tasks.inc()

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time

                worker_active_tasks.dec()

                worker_tasks_total.labels(
                    task_name=task_name,
                    status=status
                ).inc()

                worker_task_duration_seconds.labels(
                    task_name=task_name
                ).observe(duration)

        return wrapper
    return decorator


def track_database_query(operation: str, table: str):
    """Декоратор для отслеживания запросов к базе данных"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration_seconds.labels(
                    operation=operation,
                    table=table
                ).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration_seconds.labels(
                    operation=operation,
                    table=table
                ).observe(duration)

        # Возвращаем соответствующий wrapper в зависимости от типа функции
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def increment_counter(counter_name: str, labels: Dict[str, str] = None, value: float = 1):
    """Увеличение счетчика по имени"""
    counters = {
        'api_requests_total': api_requests_total,
        'anomalies_detected_total': anomalies_detected_total,
        'forecasts_generated_total': forecasts_generated_total,
        'csv_files_processed_total': csv_files_processed_total,
        'data_points_processed_total': data_points_processed_total,
        'worker_tasks_total': worker_tasks_total
    }

    counter = counters.get(counter_name)
    if counter:
        if labels:
            counter.labels(**labels).inc(value)
        else:
            counter.inc(value)
    else:
        logger.warning(f"Неизвестный счетчик: {counter_name}")


def observe_histogram(histogram_name: str, value: float, labels: Dict[str, str] = None):
    """Добавление значения в гистограмму по имени"""
    histograms = {
        'api_request_duration_seconds': api_request_duration_seconds,
        'anomaly_detection_duration_seconds': anomaly_detection_duration_seconds,
        'forecast_latency_seconds': forecast_latency_seconds,
        'csv_processing_duration_seconds': csv_processing_duration_seconds,
        'worker_task_duration_seconds': worker_task_duration_seconds,
        'database_query_duration_seconds': database_query_duration_seconds
    }

    histogram = histograms.get(histogram_name)
    if histogram:
        if labels:
            histogram.labels(**labels).observe(value)
        else:
            histogram.observe(value)
    else:
        logger.warning(f"Неизвестная гистограмма: {histogram_name}")


def set_gauge(gauge_name: str, value: float, labels: Dict[str, str] = None):
    """Установка значения gauge по имени"""
    gauges = {
        'worker_active_tasks': worker_active_tasks,
        'system_cpu_usage_percent': system_cpu_usage_percent,
        'system_memory_usage_bytes': system_memory_usage_bytes,
        'system_disk_usage_bytes': system_disk_usage_bytes,
        'database_connections_active': database_connections_active
    }

    gauge = gauges.get(gauge_name)
    if gauge:
        if labels:
            gauge.labels(**labels).set(value)
        else:
            gauge.set(value)
    else:
        logger.warning(f"Неизвестный gauge: {gauge_name}")


def get_all_metrics() -> str:
    """Получение всех метрик в формате Prometheus"""
    return metrics_collector.get_metrics()
