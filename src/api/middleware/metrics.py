"""
Prometheus метрики и middleware для FastAPI
"""

import time
from typing import Dict, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.utils import metrics as utils_metrics

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PrometheusMetrics:
    """Тонкая обертка над utils.metrics для совместимости старого кода."""

    def get_metrics(self) -> bytes:
        return utils_metrics.get_all_metrics()


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
            user_role = 'unknown'

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
            utils_metrics.increment_counter(
                'api_requests_total',
                {'method': method, 'endpoint': normalized_path, 'status_code': str(status_code), 'user_role': user_role}
            )

            utils_metrics.observe_histogram(
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
    utils_metrics.increment_counter(
        'anomalies_detected_total',
        {'equipment_type': equipment_type, 'severity': severity, 'model_name': model_name}
    )
    utils_metrics.observe_histogram('anomaly_detection_duration_seconds', duration, {'model_name': model_name, 'equipment_id': equipment_type})


def track_forecast_generation(equipment_type: str, duration: float):
    """Отслеживание генерации прогнозов"""
    utils_metrics.increment_counter('forecasts_generated_total', {'model_name': 'rms_trend_forecasting', 'equipment_id': equipment_type, 'status': 'success'})
    utils_metrics.observe_histogram('forecast_latency_seconds', duration, {'model_name': 'rms_trend_forecasting', 'equipment_id': equipment_type, 'forecast_horizon': 'auto'})


def track_signal_processing(equipment_type: str, status: str, duration: float):
    """Отслеживание обработки сигналов"""
    utils_metrics.increment_counter(
        'csv_files_processed_total',
        {'equipment_id': equipment_type, 'status': status}
    )
    utils_metrics.observe_histogram('csv_processing_duration_seconds', duration, {'equipment_id': equipment_type})


def track_file_upload(file_type: str, status: str, file_size: int):
    """Отслеживание загрузки файлов"""
    # Переадресуем на общие счетчики API при необходимости — специализированной метрики нет
    utils_metrics.increment_counter('api_requests_total', {'method': 'POST', 'endpoint': f'/upload/{file_type}', 'status_code': '200' if status=='success' else '400', 'user_role': 'system'})


def update_system_gauges(active_equipment: int, queue_size: int, db_connections: int):
    """Обновление системных gauge метрик"""
    utils_metrics.set_gauge('database_connections_active', db_connections)
