"""
Middleware для автоматического audit-логирования HTTP запросов

Этот middleware перехватывает все запросы к защищенным эндпоинтам
и автоматически логирует действия пользователей в неизменяемую таблицу audit_logs
"""

import time
from typing import Optional
from uuid import UUID

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.middleware.security import (
    SecurityAuditLogger, AuditActionType, AuditResult, enhanced_jwt_handler
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """Middleware для автоматического audit-логирования"""

    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = SecurityAuditLogger()

        # Мапинг эндпоинтов к типам действий
        self.endpoint_actions = {
            # API эндпоинты
            'POST /api/v1/upload': AuditActionType.FILE_UPLOAD,
            'GET /api/v1/signals': AuditActionType.SIGNAL_VIEW,
            'GET /api/v1/anomalies': AuditActionType.ANOMALY_REQUEST,
            'GET /api/v1/equipment': AuditActionType.EQUIPMENT_ACCESS,
            'POST /api/v1/anomalies': AuditActionType.FORECAST_REQUEST,

            # Авторизация
            'POST /auth/login': AuditActionType.LOGIN,
            'POST /auth/logout': AuditActionType.LOGOUT,
            'POST /auth/refresh': AuditActionType.TOKEN_REFRESH,

            # Админские действия
            'POST /admin': AuditActionType.ADMIN_ACTION,
            'PUT /admin': AuditActionType.ADMIN_ACTION,
            'DELETE /admin': AuditActionType.ADMIN_ACTION,
            'PATCH /admin': AuditActionType.ADMIN_ACTION,
        }

        # Эндпоинты, которые не логируются (служебные)
        self.excluded_paths = {
            '/docs', '/redoc', '/openapi.json', '/health', '/metrics'
        }

    async def dispatch(self, request: Request, call_next):
        """Обработка HTTP запроса с audit-логированием"""

        # Пропускаем служебные эндпоинты
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        start_time = time.time()

        # Извлекаем информацию о пользователе из токена (если есть)
        user_info = await self._extract_user_info(request)

        # Выполняем запрос
        response: Response = await call_next(request)

        # Определяем результат действия
        audit_result = self._determine_audit_result(response.status_code)

        # Логируем действие если пользователь аутентифицирован
        if user_info:
            await self._log_user_action(
                request=request,
                response=response,
                user_info=user_info,
                audit_result=audit_result,
                processing_time=time.time() - start_time
            )
        elif self._requires_authentication(request):
            # Логируем неавторизованный доступ к защищенным ресурсам
            await self._log_unauthorized_access(request, response)

        return response

    async def _extract_user_info(self, request: Request) -> Optional[dict]:
        """Извлечение информации о пользователе из JWT токена"""
        try:
            auth_header = request.headers.get("authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None

            token = auth_header.split(" ")[1]
            payload = enhanced_jwt_handler._decode_token(token)

            if payload.get("type") == "access":
                return {
                    'user_id': UUID(payload.get("sub")),
                    'username': payload.get("username"),
                    'role': payload.get("role"),
                    'session_id': payload.get("session_id")
                }
        except Exception:
            # Невалидный токен - просто возвращаем None (анонимный доступ)
            return None

        return None

    def _determine_audit_result(self, status_code: int) -> AuditResult:
        """Определение результата действия по HTTP статусу"""
        if 200 <= status_code < 300:
            return AuditResult.SUCCESS
        elif status_code == 401:
            return AuditResult.DENIED
        elif status_code == 403:
            return AuditResult.DENIED
        elif 400 <= status_code < 500:
            return AuditResult.FAILURE
        else:
            return AuditResult.ERROR

    def _get_action_type(self, request: Request) -> AuditActionType:
        """Определение типа действия по эндпоинту"""
        method_path = f"{request.method} {request.url.path}"

        # Прямое совпадение
        if method_path in self.endpoint_actions:
            return self.endpoint_actions[method_path]

        # Поиск по паттернам
        for pattern, action in self.endpoint_actions.items():
            if self._matches_pattern(method_path, pattern):
                return action

        # По умолчанию - доступ к оборудованию
        if '/api/v1/' in request.url.path:
            return AuditActionType.EQUIPMENT_ACCESS

        return AuditActionType.EQUIPMENT_ACCESS

    def _matches_pattern(self, method_path: str, pattern: str) -> bool:
        """Проверка соответствия пути паттерну"""
        # Простая проверка на содержание ключевых слов
        method, path = method_path.split(' ', 1)
        pattern_method, pattern_path = pattern.split(' ', 1)

        if method != pattern_method:
            return False

        # Проверка паттернов с UUID
        if '{' in pattern_path:
            pattern_parts = pattern_path.split('/')
            path_parts = path.split('/')

            if len(pattern_parts) != len(path_parts):
                return False

            for i, (pattern_part, path_part) in enumerate(zip(pattern_parts, path_parts)):
                if pattern_part.startswith('{') and pattern_part.endswith('}'):
                    # Это параметр - пропускаем
                    continue
                elif pattern_part != path_part:
                    return False

            return True

        return pattern_path in path

    def _requires_authentication(self, request: Request) -> bool:
        """Проверка, требует ли эндпоинт аутентификации"""
        protected_paths = ['/api/v1/', '/admin/']
        return any(request.url.path.startswith(path) for path in protected_paths)

    def _extract_resource_info(self, request: Request) -> tuple[Optional[str], Optional[UUID], Optional[str]]:
        """Извлечение информации о ресурсе из URL"""
        path_parts = request.url.path.split('/')

        # Определяем тип ресурса
        if 'signals' in path_parts:
            resource_type = 'signal'
        elif 'anomalies' in path_parts:
            resource_type = 'equipment'
        elif 'equipment' in path_parts:
            resource_type = 'equipment'
        elif 'upload' in path_parts:
            resource_type = 'file'
        else:
            resource_type = None

        # Ищем UUID в пути
        resource_id = None
        for part in path_parts:
            try:
                resource_id = UUID(part)
                break
            except ValueError:
                continue

        # Название ресурса (можно расширить логику)
        resource_name = None
        if resource_id:
            resource_name = f"{resource_type}_{str(resource_id)[:8]}"

        return resource_type, resource_id, resource_name

    async def _log_user_action(
        self,
        request: Request,
        response: Response,
        user_info: dict,
        audit_result: AuditResult,
        processing_time: float
    ):
        """Логирование действия аутентифицированного пользователя"""

        action_type = self._get_action_type(request)
        resource_type, resource_id, resource_name = self._extract_resource_info(request)

        # Дополнительные данные
        additional_data = {
            'processing_time_ms': round(processing_time * 1000, 2),
            'response_status': response.status_code,
            'response_size': len(response.body) if hasattr(response, 'body') else None
        }

        # Специальная обработка для критичных действий
        if action_type in [AuditActionType.ADMIN_ACTION, AuditActionType.SYSTEM_CONFIG_CHANGE]:
            additional_data['critical_action'] = True

        await self.audit_logger.log_action(
            user_id=user_info['user_id'],
            username=user_info['username'],
            user_role=user_info['role'],
            action_type=action_type,
            result=audit_result,
            request=request,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            additional_data=additional_data,
            session_id=user_info.get('session_id')
        )

    async def _log_unauthorized_access(self, request: Request, response: Response):
        """Логирование неавторизованного доступа к защищенным ресурсам"""

        # Используем анонимного пользователя для логирования
        anonymous_user_id = UUID('00000000-0000-0000-0000-000000000000')

        await self.audit_logger.log_action(
            user_id=anonymous_user_id,
            username="anonymous",
            user_role="none",
            action_type=AuditActionType.PERMISSION_DENIED,
            result=AuditResult.DENIED,
            request=request,
            action_description=f"Unauthorized access attempt to {request.method} {request.url.path}",
            additional_data={
                'response_status': response.status_code,
                'anonymous_access': True
            }
        )
