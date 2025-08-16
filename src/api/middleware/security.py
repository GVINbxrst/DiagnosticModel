"""
Расширенная система безопасности и audit-логирования для FastAPI

Включает:
- Усиленная JWT система с session management
- Audit-логирование всех действий пользователей
- Неизменяемое хранение логов безопасности
- Мониторинг подозрительной активности
"""

import hashlib
import secrets
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from enum import Enum
import ipaddress

from fastapi import Request, HTTPException, status
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from src.database.connection import get_async_session
from src.database.models import User
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AuditActionType(str, Enum):
    """Типы действий для audit-логирования"""
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    FILE_UPLOAD = "file_upload"
    SIGNAL_VIEW = "signal_view"
    ANOMALY_REQUEST = "anomaly_request"
    FORECAST_REQUEST = "forecast_request"
    EQUIPMENT_ACCESS = "equipment_access"
    ADMIN_ACTION = "admin_action"
    DATA_EXPORT = "data_export"
    MODEL_TRAINING = "model_training"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    USER_MANAGEMENT = "user_management"
    FAILED_LOGIN = "failed_login"
    PERMISSION_DENIED = "permission_denied"


class AuditResult(str, Enum):
    """Результаты действий для audit-логирования"""
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    ERROR = "error"


class SecurityAuditLogger:
    """Класс для audit-логирования действий пользователей"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def log_action(
        self,
        user_id: UUID,
        username: str,
        user_role: str,
        action_type: AuditActionType,
        result: AuditResult,
        request: Request,
        action_description: str = None,
        resource_type: str = None,
        resource_id: UUID = None,
        resource_name: str = None,
        additional_data: Dict = None,
        session_id: str = None
    ):
        """
        Логирование действия пользователя в неизменяемую таблицу audit_logs

        Args:
            user_id: ID пользователя
            username: Имя пользователя
            user_role: Роль пользователя
            action_type: Тип действия
            result: Результат действия
            request: FastAPI Request объект
            action_description: Описание действия
            resource_type: Тип ресурса (equipment, signal, model, etc.)
            resource_id: ID ресурса
            resource_name: Название ресурса
            additional_data: Дополнительные данные
            session_id: ID сессии
        """

        try:
            async with get_async_session() as session:
                # Извлекаем информацию из запроса
                client_ip = self._get_client_ip(request)
                user_agent = request.headers.get("user-agent", "")
                endpoint = str(request.url.path)
                http_method = request.method

                # Подготавливаем данные запроса (исключаем чувствительную информацию)
                request_data = self._sanitize_request_data(request)

                # Формируем описание действия
                if not action_description:
                    action_description = self._generate_action_description(
                        action_type, http_method, endpoint, resource_type
                    )

                # SQL запрос для вставки в audit_logs
                audit_query = """
                    INSERT INTO security.audit_logs (
                        user_id, username, user_role, action_type, action_description,
                        result, endpoint, http_method, request_ip, user_agent,
                        request_data, resource_type, resource_id, resource_name,
                        session_id, additional_data
                    ) VALUES (
                        :user_id, :username, :user_role, :action_type, :action_description,
                        :result, :endpoint, :http_method, :request_ip, :user_agent,
                        :request_data, :resource_type, :resource_id, :resource_name,
                        :session_id, :additional_data
                    )
                """

                await session.execute(audit_query, {
                    'user_id': str(user_id),
                    'username': username,
                    'user_role': user_role,
                    'action_type': action_type.value,
                    'action_description': action_description,
                    'result': result.value,
                    'endpoint': endpoint,
                    'http_method': http_method,
                    'request_ip': client_ip,
                    'user_agent': user_agent,
                    'request_data': request_data,
                    'resource_type': resource_type,
                    'resource_id': str(resource_id) if resource_id else None,
                    'resource_name': resource_name,
                    'session_id': session_id,
                    'additional_data': additional_data
                })

                await session.commit()

                # Также логируем в стандартный лог
                self.logger.info(
                    f"AUDIT: {username} ({user_role}) {action_type.value} - {result.value}",
                    extra={
                        'audit_action': action_type.value,
                        'audit_result': result.value,
                        'user_id': str(user_id),
                        'username': username,
                        'user_role': user_role,
                        'endpoint': endpoint,
                        'client_ip': client_ip,
                        'resource_type': resource_type,
                        'resource_id': str(resource_id) if resource_id else None
                    }
                )

        except Exception as e:
            # Критическая ошибка - не можем записать audit лог
            self.logger.error(f"CRITICAL: Failed to write audit log: {e}", exc_info=True)
            # Не поднимаем исключение, чтобы не нарушить работу основного приложения

    def _get_client_ip(self, request: Request) -> str:
        """Получение IP адреса клиента с учетом прокси"""
        # Проверяем заголовки прокси
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Берем первый IP из списка
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback на прямое подключение
        return request.client.host if request.client else "unknown"

    def _sanitize_request_data(self, request: Request) -> Dict:
        """Очистка данных запроса от чувствительной информации"""
        sensitive_keys = {
            'password', 'token', 'authorization', 'secret', 'key',
            'refresh_token', 'access_token', 'api_key'
        }

        data = {}

        # Query параметры
        if request.query_params:
            data['query_params'] = {
                k: v if k.lower() not in sensitive_keys else "[REDACTED]"
                for k, v in request.query_params.items()
            }

        # Headers (исключаем чувствительные)
        if request.headers:
            data['headers'] = {
                k: v if k.lower() not in sensitive_keys else "[REDACTED]"
                for k, v in request.headers.items()
                if k.lower() not in {'authorization', 'cookie'}
            }

        return data

    def _generate_action_description(
        self,
        action_type: AuditActionType,
        method: str,
        endpoint: str,
        resource_type: str = None
    ) -> str:
        """Генерация описания действия"""
        descriptions = {
            AuditActionType.LOGIN: "User logged in",
            AuditActionType.LOGOUT: "User logged out",
            AuditActionType.TOKEN_REFRESH: "Refreshed access token",
            AuditActionType.FILE_UPLOAD: "Uploaded CSV file",
            AuditActionType.SIGNAL_VIEW: f"Viewed signal data via {method} {endpoint}",
            AuditActionType.ANOMALY_REQUEST: f"Requested anomaly data via {method} {endpoint}",
            AuditActionType.FORECAST_REQUEST: f"Requested forecast data via {method} {endpoint}",
            AuditActionType.EQUIPMENT_ACCESS: f"Accessed equipment data via {method} {endpoint}",
            AuditActionType.ADMIN_ACTION: f"Performed admin action: {method} {endpoint}",
            AuditActionType.FAILED_LOGIN: "Failed login attempt",
            AuditActionType.PERMISSION_DENIED: f"Permission denied for {method} {endpoint}"
        }

        base_description = descriptions.get(action_type, f"Action {action_type.value}")

        if resource_type:
            base_description += f" (Resource: {resource_type})"

        return base_description


class EnhancedJWTHandler:
    """Расширенный обработчик JWT с session management"""

    def __init__(self):
        from src.config.settings import get_settings
        settings = get_settings()

        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        self.audit_logger = SecurityAuditLogger()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    async def create_session(
        self,
        user: User,
        request: Request
    ) -> Dict[str, Any]:
        """
        Создание новой пользовательской сессии с токенами

        Args:
            user: Пользователь
            request: HTTP запрос

        Returns:
            Данные сессии с токенами
        """
        session_id = str(uuid4())

        # Создаем токены
        access_token = self._create_access_token(user, session_id)
        refresh_token = self._create_refresh_token(user, session_id)

        # Сохраняем сессию в БД
        await self._save_session(user, session_id, refresh_token, request)

        # Логируем успешный вход
        role_value = getattr(user.role, 'value', user.role)
        await self.audit_logger.log_action(
            user_id=user.id,
            username=user.username,
            user_role=role_value,
            action_type=AuditActionType.LOGIN,
            result=AuditResult.SUCCESS,
            request=request,
            session_id=session_id
        )

        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'bearer',
            'expires_in': self.access_token_expire * 60,
            'session_id': session_id
        }

    async def refresh_session(
        self,
        refresh_token: str,
        request: Request
    ) -> Dict[str, Any]:
        """
        Обновление сессии по refresh токену

        Args:
            refresh_token: Refresh токен
            request: HTTP запрос

        Returns:
            Новые токены
        """
        # Декодируем refresh токен
        payload = self._decode_token(refresh_token)

        if payload.get('type') != 'refresh':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token type"
            )

        user_id = UUID(payload.get('sub'))
        session_id = payload.get('session_id')

        # Проверяем сессию в БД
        async with get_async_session() as db_session:
            session_query = select(User).where(User.id == user_id)
            result = await db_session.execute(session_query)
            user = result.scalar_one_or_none()

            if not user or not user.is_active:
                await self.audit_logger.log_action(
                    user_id=user_id,
                    username=user.username if user else "unknown",
                    user_role=getattr(user.role, 'value', user.role) if user else "unknown",
                    action_type=AuditActionType.TOKEN_REFRESH,
                    result=AuditResult.FAILURE,
                    request=request,
                    action_description="Refresh token for inactive/deleted user"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or inactive"
                )

            # Проверяем активность сессии
            if not await self._is_session_active(session_id, refresh_token):
                await self.audit_logger.log_action(
                    user_id=user.id,
                    username=user.username,
                    user_role=getattr(user.role, 'value', user.role),
                    action_type=AuditActionType.TOKEN_REFRESH,
                    result=AuditResult.FAILURE,
                    request=request,
                    action_description="Attempted to use revoked or expired session"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired or revoked"
                )

        # Создаем новые токены
        new_access_token = self._create_access_token(user, session_id)
        new_refresh_token = self._create_refresh_token(user, session_id)

        # Обновляем сессию в БД
        await self._update_session(session_id, new_refresh_token, request)

        # Логируем обновление токена
        await self.audit_logger.log_action(
            user_id=user.id,
            username=user.username,
            user_role=getattr(user.role, 'value', user.role),
            action_type=AuditActionType.TOKEN_REFRESH,
            result=AuditResult.SUCCESS,
            request=request,
            session_id=session_id
        )

        return {
            'access_token': new_access_token,
            'refresh_token': new_refresh_token,
            'token_type': 'bearer',
            'expires_in': self.access_token_expire * 60
        }

    async def revoke_session(
        self,
        session_id: str,
        reason: str = "User logout"
    ):
        """Отзыв сессии"""
        async with get_async_session() as session:
            revoke_query = """
                UPDATE security.user_sessions 
                SET is_active = FALSE, 
                    revoked_at = NOW(),
                    revoked_reason = :reason
                WHERE session_token = :session_id AND is_active = TRUE
            """

            await session.execute(revoke_query, {
                'session_id': session_id,
                'reason': reason
            })
            await session.commit()

    async def cleanup_expired_sessions(self) -> int:
        """Очистка истекших сессий"""
        async with get_async_session() as session:
            cleanup_query = "SELECT security.cleanup_expired_sessions()"
            result = await session.execute(cleanup_query)
            count = result.scalar()
            await session.commit()

            self.logger.info(f"Cleaned up {count} expired sessions")
            return count

    def _create_access_token(self, user: User, session_id: str) -> str:
        """Создание access токена"""
        import jwt
        expire = datetime.now(UTC) + timedelta(minutes=self.access_token_expire)
        role_value = user.role.value if hasattr(user.role, 'value') else str(user.role)
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": role_value,
            "session_id": session_id,
            "type": "access",
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid4())  # JWT ID для отслеживания
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def _create_refresh_token(self, user: User, session_id: str) -> str:
        """Создание refresh токена"""
        import jwt
        expire = datetime.now(UTC) + timedelta(days=self.refresh_token_expire)
        payload = {
            "sub": str(user.id),
            "session_id": session_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(UTC),
            "jti": str(uuid4())
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def _decode_token(self, token: str) -> Dict[str, Any]:
        """Декодирование токена"""
        import jwt

        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    async def _save_session(
        self,
        user: User,
        session_id: str,
        refresh_token: str,
        request: Request
    ):
        """Сохранение сессии в БД"""
        async with get_async_session() as session:
            # Хешируем refresh токен для безопасного хранения
            refresh_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

            insert_query = """
                INSERT INTO security.user_sessions (
                    user_id, session_token, refresh_token_hash,
                    expires_at, ip_address, user_agent
                ) VALUES (
                    :user_id, :session_token, :refresh_hash,
                    :expires_at, :ip_address, :user_agent
                )
            """

            client_ip = self.audit_logger._get_client_ip(request)
            user_agent = request.headers.get("user-agent", "")
            expires_at = datetime.now(UTC) + timedelta(days=self.refresh_token_expire)

            await session.execute(insert_query, {
                'user_id': str(user.id),
                'session_token': session_id,
                'refresh_hash': refresh_hash,
                'expires_at': expires_at,
                'ip_address': client_ip,
                'user_agent': user_agent
            })
            await session.commit()

    async def _update_session(
        self,
        session_id: str,
        new_refresh_token: str,
        request: Request
    ):
        """Обновление сессии"""
        async with get_async_session() as session:
            refresh_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()
            client_ip = self.audit_logger._get_client_ip(request)

            update_query = """
                UPDATE security.user_sessions 
                SET refresh_token_hash = :refresh_hash,
                    last_activity = NOW(),
                    ip_address = :ip_address
                WHERE session_token = :session_id AND is_active = TRUE
            """

            await session.execute(update_query, {
                'refresh_hash': refresh_hash,
                'ip_address': client_ip,
                'session_id': session_id
            })
            await session.commit()

    async def _is_session_active(self, session_id: str, refresh_token: str) -> bool:
        """Проверка активности сессии"""
        async with get_async_session() as session:
            refresh_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

            check_query = """
                SELECT COUNT(*) FROM security.user_sessions 
                WHERE session_token = :session_id 
                    AND refresh_token_hash = :refresh_hash
                    AND is_active = TRUE 
                    AND expires_at > NOW()
            """

            result = await session.execute(check_query, {
                'session_id': session_id,
                'refresh_hash': refresh_hash
            })

            return result.scalar() > 0


# Глобальный экземпляр расширенного JWT обработчика
enhanced_jwt_handler = EnhancedJWTHandler()

# Глобальный экземпляр audit логгера
audit_logger = SecurityAuditLogger()
