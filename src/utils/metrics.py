# Метрики Prometheus: сбор/экспорт для мониторинга

import time
from functools import wraps
from typing import Dict, Optional, Callable, Any
import asyncio
from datetime import datetime
import inspect

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
def increment_counter(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для increment_counter")
        return
    if labels:
        metric.labels(**labels).inc(value)
    else:
        metric.inc(value)

def observe_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для observe_histogram")
        return
    import inspect
    import asyncio

    if labels:
        metric.labels(**labels).observe(value)
    else:
        metric.observe(value)

def set_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    # Установить значение gauge
    metric = globals().get(name)
    if metric is None:
        logger.warning(f"Метрика {name} не найдена для set_gauge")
        return
    try:
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    except Exception as e:
        logger.warning(f"Ошибка при установке gauge {name}: {e}")

def increment(name: str, labels: Optional[Dict[str, str]] = None):
    # Alias increment_counter (совместимость)
    return increment_counter(name, labels)

def observe(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    # Alias observe_hist (совместимость)
    return observe_histogram(name, value, labels)

def track_worker_task(task_name: str, status: str, duration: Optional[float] = None):
    # Учёт метрик задач воркера
    try:
        increment_counter('worker_tasks_total', {'task_name': task_name, 'status': status})
        if duration is not None:
            observe_histogram('worker_task_duration_seconds', duration, {'task_name': task_name})
    except Exception as e:
        logger.debug(f"Не удалось трекать метрики задачи {task_name}: {e}")

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


## Удалены дублирующие определения set_gauge/track_worker_task (оставлены ранние версии выше)

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

# Перетренировки и версии моделей
model_retrain_total = Counter(
    'model_retrain_total',
    'Количество запусков переобучения моделей',
    ['model_group', 'status'],
    registry=REGISTRY
)

model_version_gauge = Gauge(
    'model_version_info',
    'Текущая активная версия моделей (label version -> 1)',
    ['model_group', 'version'],
    registry=REGISTRY
)

# Качество переобучения аномалий
anomaly_retrain_score_p98 = Gauge(
    'anomaly_retrain_score_p98',
    '98-й перцентиль внутренних score при обучении stream модели',
    ['model_type'],
    registry=REGISTRY
)
anomaly_retrain_outlier_ratio = Gauge(
    'anomaly_retrain_outlier_ratio',
    'Доля точек превысивших новый threshold во время retrain (оценка контаминации)',
    ['model_type'],
    registry=REGISTRY
)

# Кластеризация
clustering_runs_total = Counter(
    'clustering_runs_total',
    'Количество запусков пайплайна кластеризации',
    ['status'],
    registry=REGISTRY
)
clustering_last_duration_seconds = Gauge(
    'clustering_last_duration_seconds',
    'Длительность последнего запуска кластеризации (сек)',
    registry=REGISTRY
)

# --- Метрики готовности данных / обучающего датасета ---
features_with_embedding_total = Gauge(
    'features_with_embedding_total',
    'Количество Feature записей с embedding',
    registry=REGISTRY
)
embedding_coverage_ratio = Gauge(
    'embedding_coverage_ratio',
    'Доля features с embedding (0..1)',
    registry=REGISTRY
)
clustering_noise_ratio = Gauge(
    'clustering_noise_ratio',
    'Доля точек без присвоенного cluster_id (noise)',
    registry=REGISTRY
)
clustering_label_coverage_ratio = Gauge(
    'clustering_label_coverage_ratio',
    'Доля кластеров, имеющих назначенную метку дефекта',
    registry=REGISTRY
)
min_cluster_size_labeled = Gauge(
    'min_cluster_size_labeled',
    'Минимальный размер среди размеченных кластеров',
    registry=REGISTRY
)
p50_cluster_size = Gauge(
    'p50_cluster_size',
    'Медианный размер кластеров',
    registry=REGISTRY
)
p90_cluster_size = Gauge(
    'p90_cluster_size',
    '90-й перцентиль размеров кластеров',
    registry=REGISTRY
)
data_recency_seconds = Gauge(
    'data_recency_seconds',
    'Сколько секунд прошло с момента последнего window_end Feature',
    registry=REGISTRY
)
clustering_drift_score = Gauge(
    'clustering_drift_score',
    'Оценка дрейфа распределения кластеров (Jensen-Shannon divergence 0..1)',
    registry=REGISTRY
)
training_snapshot_timestamp = Gauge(
    'training_snapshot_timestamp',
    'Unix-время (UTC) последнего успешного snapshot обучающего датасета',
    registry=REGISTRY
)

# CSV ingestion
csv_ingest_files_total = Counter(
    'csv_ingest_files_total',
    'Количество CSV файлов, обработанных лоадером',
    ['status'],  # success|error
    registry=REGISTRY
)
csv_ingest_rows_total = Counter(
    'csv_ingest_rows_total',
    'Суммарное количество строк успешно загруженных CSV',
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

forecast_tasks_total = Counter(
    'forecast_tasks_total',
    'Количество запусков фоновой задачи прогнозирования',
    ['task', 'status'],
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

# Метрики пачек CSV
csv_batch_rows = Histogram(
    'csv_batch_rows',
    'Размер обработанных пачек CSV (в строках)',
    ['equipment_id'],
    buckets=[100, 500, 1000, 5000, 10000, 20000],
    registry=REGISTRY
)

csv_batch_duration_seconds = Histogram(
    'csv_batch_duration_seconds',
    'Время обработки одной пачки CSV (секунды)',
    ['equipment_id'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
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

# Кеш моделей (hits / misses / store / invalidate)
model_cache_events_total = Counter(
    'model_cache_events_total',
    'События кеша ML моделей',
    ['model_name', 'event'],  # hit|miss|store|invalidate
    registry=REGISTRY
)

# Дедупликация CSV
csv_duplicates_total = Counter(
    'csv_duplicates_total',
    'Количество попыток загрузить уже существующий CSV (по file_hash)',
    ['equipment_id'],
    registry=REGISTRY
)
csv_duplicate_skipped_total = Counter(
    'csv_duplicate_skipped_total',
    'Количество пропущенных CSV из-за дубликата (без повторной обработки)',
    ['equipment_id'],
    registry=REGISTRY
)

# Метрики Feature Store
feature_store_hit_total = Counter(
    'feature_store_hit_total',
    'Количество успешных чтений из Feature Store (без fallback)',
    ['backend'],
    registry=REGISTRY
)

feature_store_miss_total = Counter(
    'feature_store_miss_total',
    'Количество промахов Feature Store (fallback либо пустой результат)',
    ['backend', 'reason'],
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
    # Коллектор системных метрик

    def __init__(self):
        self.start_time = time.time()
        self.update_app_info()

    def update_app_info(self):
        app_info.info({
            'version': '1.0.0',
            'environment': 'development',
            'started_at': datetime.now().isoformat()
        })

    def update_system_metrics(self):
        # Обновить системные метрики
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage_percent.set(cpu_percent)

            mem = psutil.virtual_memory()
            system_memory_usage_bytes.labels(type='used').set(mem.used)
            system_memory_usage_bytes.labels(type='available').set(mem.available)
            system_memory_usage_bytes.labels(type='total').set(mem.total)

            disk = psutil.disk_usage('/')
            system_disk_usage_bytes.labels(path='/', type='used').set(disk.used)
            system_disk_usage_bytes.labels(path='/', type='free').set(disk.free)
            system_disk_usage_bytes.labels(path='/', type='total').set(disk.total)
        except Exception as e:
            logger.debug(f"Не удалось обновить системные метрики: {e}")

    def get_metrics(self) -> bytes:
        # Получить все метрики
        return generate_latest(REGISTRY)

    # --- Унифицированные утилиты ---

metrics_collector = MetricsCollector()


def get_all_metrics() -> bytes:
    """Совместимый alias для экспорта всех метрик."""
    return generate_latest(REGISTRY)


# --- Алиасы для совместимости тестов/старого кода ---
def increment(name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
    return increment_counter(name, labels, value)


def observe(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    return observe_histogram(name, value, labels)


__all__ = [
    'REGISTRY',
    'api_requests_total', 'api_request_duration_seconds',
    'anomalies_detected_total', 'anomaly_detection_duration_seconds',
    'forecast_latency_seconds', 'forecasts_generated_total',
    'csv_files_processed_total', 'csv_processing_duration_seconds', 'data_points_processed_total',
    'worker_tasks_total', 'worker_task_duration_seconds', 'worker_active_tasks',
    'system_cpu_usage_percent', 'system_memory_usage_bytes', 'system_disk_usage_bytes',
    'database_connections_active', 'database_query_duration_seconds', 'app_info',
    'increment_counter', 'observe_histogram', 'set_gauge', 'get_metrics', 'get_all_metrics', 'observe_latency',
    'metrics_collector',
    'increment', 'observe', 'track_worker_task', 'safe_add'
]


async def safe_add(session, instance):
    """Безопасно вызвать session.add для реальной сессии или моков.

    Цели:
    - Не вызывать предупреждения об не-await coroutine если add замокан через AsyncMock
    - Унифицированно поддерживать синхронный add (реальный AsyncSession) и асинхронный (мок)
    """
    add_attr = getattr(session, 'add', None)
    if add_attr is None:
        logger.debug('safe_add: у сессии отсутствует метод add')
        return
    try:
        res = add_attr(instance)
        if inspect.isawaitable(res):
            await res
    except Exception as e:  # pragma: no cover - защитный слой
        logger.debug(f'safe_add: ошибка при добавлении объекта: {e}')
        raise
