"""
Prometheus метрики и middleware для FastAPI
"""

import time
from typing import Dict, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrometheusMetrics:
    """Класс для управления Prometheus метриками"""

    def __init__(self):
        # Инициализируем метрики
        self.init_metrics()

    def init_metrics(self):
        """Инициализация всех метрик"""

        # HTTP запросы
        self.api_requests_total = Counter(
            'api_requests_total',
            'Общее количество HTTP запросов',
            ['method', 'endpoint', 'status_code', 'user_role']
        )

        self.api_request_duration_seconds = Histogram(
            'api_request_duration_seconds',
            'Время выполнения HTTP запросов',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        # Аномалии
        self.anomalies_detected_total = Counter(
            'anomalies_detected_total',
            'Общее количество обнаруженных аномалий',
            ['equipment_type', 'severity', 'model_name']
        )

        self.anomaly_detection_duration_seconds = Histogram(
            'anomaly_detection_duration_seconds',
            'Время выполнения детекции аномалий',
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        )

        # Прогнозирование
        self.forecast_latency_seconds = Histogram(
            'forecast_latency_seconds',
            'Время выполнения прогнозирования',
            buckets=[5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
        )

        self.forecasts_generated_total = Counter(
            'forecasts_generated_total',
            'Общее количество сгенерированных прогнозов',
            ['equipment_type']
        )

        # Обработка данных
        self.signals_processed_total = Counter(
            'signals_processed_total',
            'Общее количество обработанных сигналов',
            ['equipment_type', 'status']
        )

        self.signal_processing_duration_seconds = Histogram(
            'signal_processing_duration_seconds',
            'Время обработки сигналов',
            buckets=[10.0, 30.0, 60.0, 300.0, 600.0, 1800.0]
        )

        # Состояние системы
        self.active_equipment_count = Gauge(
            'active_equipment_count',
            'Количество активного оборудования'
        )

        self.processing_queue_size = Gauge(
            'processing_queue_size',
            'Размер очереди обработки'
        )

        self.database_connections_active = Gauge(
            'database_connections_active',
            'Количество активных подключений к БД'
        )

        # Ошибки
        self.api_errors_total = Counter(
            'api_errors_total',
            'Общее количество ошибок API',
            ['status_code', 'error_type']
        )

        self.worker_task_failures_total = Counter(
            'worker_task_failures_total',
            'Количество неудачных worker задач',
            ['task_name', 'error_type']
        )

        # Файлы
        self.files_uploaded_total = Counter(
            'files_uploaded_total',
            'Общ��е количество загруженных файлов',
            ['file_type', 'status']
        )

        self.file_upload_size_bytes = Histogram(
            'file_upload_size_bytes',
            'Размер загружаемых файлов в байт��х',
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600]  # 1KB - 100MB
        )

        logger.info("📊 Prometheus метрики инициализированы")

    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None):
        """Увеличение счетчика"""
        try:
            metric = getattr(self, metric_name)
            if labels:
                metric.labels(**labels).inc()
            else:
                metric.inc()
        except AttributeError:
            logger.warning(f"Метрика {metric_name} не найдена")

    def observe_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Добавление значения в гистограмму"""
        try:
            metric = getattr(self, metric_name)
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        except AttributeError:
            logger.warning(f"Метрика {metric_name} не найдена")

    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Установка значения gauge"""
        try:
            metric = getattr(self, metric_name)
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        except AttributeError:
            logger.warning(f"Метрика {metric_name} не найдена")

    def get_metrics(self) -> str:
        """Получение всех метрик в формате Prometheus"""
        return generate_latest()


# Глобальный экземпляр метрик
metrics = PrometheusMetrics()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware для сбора метрик HTTP запросов"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Извлекаем информацию о запросе
        method = request.method
        path = request.url.path
        user_role = 'unknown'
        user_id = 'anonymous'

        # Пытаемся получить информацию о пользователе из токена
        try:
            auth_header = request.headers.get('authorization')
            if auth_header and hasattr(request.state, 'user'):
                user_role = getattr(request.state.user, 'role', 'unknown')
                user_id = getattr(request.state.user, 'id', 'unknown')
        except Exception:
            # Игнорируем ошибки извлечения пользователя для метрик
            ...

        response = None
        status_code = 500

        try:
            # Выполняем запрос
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Вычисляем время выполнения
            duration = time.time() - start_time

            # Нормализуем путь для метрик (убираем ID и параметры)
            normalized_path = self._normalize_path(path)

            # Обновляем метрики
            metrics.increment_counter(
                'api_requests_total',
                {'method': method, 'endpoint': normalized_path, 'status_code': str(status_code), 'user_role': user_role}
            )

            metrics.observe_histogram(
                'api_request_duration_seconds',
                duration,
                {'method': method, 'endpoint': normalized_path}
            )

            # Логируем медленные запросы
            if duration > 5.0:
                logger.warning(f"Медленный запрос: {method} {normalized_path} выполнялся {duration:.2f}s")

    def _normalize_path(self, path: str) -> str:
        """Нормализация пути для группировки метрик"""
        # Заменяем числовые ID на {id}
        import re
        normalized = re.sub(r'/\d+', '/{id}', path)

        # Группируем похожие endpoints
        if '/equipment/' in normalized and '/files' in normalized:
            return '/equipment/{id}/files'
        elif '/signals/' in normalized:
            return '/signals/{id}'
        elif '/anomalies/' in normalized:
            return '/anomalies/{id}'
        elif '/features/' in normalized:
            return '/features/{id}'

        return normalized

    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента"""
        # Проверяем заголовки прокси
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()

        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip

        # Возвращаем IP из соединения
        if hasattr(request.client, 'host'):
            return request.client.host

        return 'unknown'


# Дополнительные функции для метрик
def track_anomaly_detection(equipment_type: str, severity: str, model_name: str, duration: float):
    """Отслеживание детекции аномалий"""
    metrics.increment_counter(
        'anomalies_detected_total',
        {'equipment_type': equipment_type, 'severity': severity, 'model_name': model_name}
    )
    metrics.observe_histogram('anomaly_detection_duration_seconds', duration)


def track_forecast_generation(equipment_type: str, duration: float):
    """Отслеживание генерации прогнозов"""
    metrics.increment_counter('forecasts_generated_total', {'equipment_type': equipment_type})
    metrics.observe_histogram('forecast_latency_seconds', duration)


def track_signal_processing(equipment_type: str, status: str, duration: float):
    """Отслеживание обработки сигналов"""
    metrics.increment_counter(
        'signals_processed_total',
        {'equipment_type': equipment_type, 'status': status}
    )
    metrics.observe_histogram('signal_processing_duration_seconds', duration)


def track_file_upload(file_type: str, status: str, file_size: int):
    """Отслеживание загрузки файлов"""
    metrics.increment_counter('files_uploaded_total', {'file_type': file_type, 'status': status})
    metrics.observe_histogram('file_upload_size_bytes', file_size)


def update_system_gauges(active_equipment: int, queue_size: int, db_connections: int):
    """Обновление системных gauge метрик"""
    metrics.set_gauge('active_equipment_count', active_equipment)
    metrics.set_gauge('processing_queue_size', queue_size)
    metrics.set_gauge('database_connections_active', db_connections)
