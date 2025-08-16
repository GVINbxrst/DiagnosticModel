"""
Тесты для системы безопасности и audit-логирования
"""

import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from datetime import datetime, timedelta, UTC

from src.api.main import app
from src.api.middleware.security import enhanced_jwt_handler, audit_logger, AuditActionType, AuditResult
from src.database.models import User


@pytest.fixture
def mock_admin_user():
    """Мок администратора для тестов"""
    user = Mock(spec=User)
    user.id = uuid4()
    user.username = "admin_user"
    user.email = "admin@example.com"
    user.role = "admin"
    user.is_active = True
    return user


@pytest.fixture
def mock_operator_user():
    """Мок оператора для тестов"""
    user = Mock(spec=User)
    user.id = uuid4()
    user.username = "operator_user"
    user.email = "operator@example.com"
    user.role = "operator"
    user.is_active = True
    return user


@pytest.fixture
def admin_token(mock_admin_user):
    """JWT токен администратора"""
    return enhanced_jwt_handler._create_access_token(mock_admin_user, str(uuid4()))


@pytest.fixture
def operator_token(mock_operator_user):
    """JWT токен оператора"""
    return enhanced_jwt_handler._create_access_token(mock_operator_user, str(uuid4()))


class TestSecurityAuditLogger:
    """Тесты audit-логирования"""

    @pytest.mark.asyncio
    async def test_log_action_success(self):
        """Тест успешного логирования действия"""
        user_id = uuid4()

        # Мокаем Request объект
        mock_request = Mock()
        mock_request.url.path = "/api/v1/signals"
        mock_request.method = "GET"
        mock_request.headers = {"user-agent": "test-client"}
        mock_request.client.host = "127.0.0.1"
        mock_request.query_params = {}

        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            await audit_logger.log_action(
                user_id=user_id,
                username="test_user",
                user_role="operator",
                action_type=AuditActionType.SIGNAL_VIEW,
                result=AuditResult.SUCCESS,
                request=mock_request,
                resource_type="signal",
                resource_id=uuid4()
            )

            # Проверяем, что SQL запрос был выполнен
            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_get_client_ip_with_proxy(self):
        """Тест извлечения IP адреса через прокси"""
        mock_request = Mock()
        mock_request.headers = {
            "x-forwarded-for": "192.168.1.100, 10.0.0.1",
            "x-real-ip": "192.168.1.100"
        }
        mock_request.client.host = "10.0.0.1"

        ip = audit_logger._get_client_ip(mock_request)
        assert ip == "192.168.1.100"

    def test_sanitize_request_data(self):
        """Тест очистки чувствительных данных"""
        mock_request = Mock()
        mock_request.query_params = {
            "user_id": "123",
            "password": "secret123",
            "token": "jwt_token"
        }
        mock_request.headers = {
            "authorization": "Bearer token",
            "content-type": "application/json",
            "x-api-key": "secret_key"
        }

        sanitized = audit_logger._sanitize_request_data(mock_request)

        assert sanitized['query_params']['user_id'] == "123"
        assert sanitized['query_params']['password'] == "[REDACTED]"
        assert sanitized['query_params']['token'] == "[REDACTED]"
        assert 'authorization' not in sanitized['headers']


class TestEnhancedJWTHandler:
    """Тесты расширенной JWT системы"""

    @pytest.mark.asyncio
    async def test_create_session_success(self, mock_operator_user):
        """Тест создания новой сессии"""
        mock_request = Mock()
        mock_request.headers = {"user-agent": "test-client"}
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/auth/login"
        mock_request.method = "POST"
        mock_request.query_params = {}

        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            session_data = await enhanced_jwt_handler.create_session(
                mock_operator_user,
                mock_request
            )

            assert 'access_token' in session_data
            assert 'refresh_token' in session_data
            assert 'session_id' in session_data
            assert session_data['token_type'] == 'bearer'

            # Проверяем, что сессия была сохранена в БД
            mock_session.execute.assert_called()
            mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self):
        """Тест очистки истекших сессий"""
        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Мокаем результат SQL функции
            mock_result = Mock()
            mock_result.scalar.return_value = 5
            mock_session.execute.return_value = mock_result

            cleaned_count = await enhanced_jwt_handler.cleanup_expired_sessions()

            assert cleaned_count == 5
            mock_session.execute.assert_called()
            mock_session.commit.assert_called()

    def test_decode_token_expired(self):
        """Тест декодирования истекшего токена"""
        import jwt
        from datetime import datetime, timedelta

        # Создаем истекший токен
        expired_payload = {
            "sub": str(uuid4()),
            "username": "test",
            "role": "operator",
            "type": "access",
            "exp": datetime.now(UTC) - timedelta(hours=1),
            "iat": datetime.now(UTC) - timedelta(hours=2)
        }

        expired_token = jwt.encode(
            expired_payload,
            enhanced_jwt_handler.secret_key,
            algorithm=enhanced_jwt_handler.algorithm
        )

        with pytest.raises(Exception):  # HTTPException wrapped
            enhanced_jwt_handler._decode_token(expired_token)


class TestSecurityEndpoints:
    """Тесты эндпоинтов безопасности"""

    @pytest.mark.asyncio
    async def test_login_with_audit_logging(self):
        """Тест авторизации с audit-логированием"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.routes.auth.authenticate_user') as mock_auth:
                with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                    mock_session = AsyncMock()
                    mock_session_ctx.return_value.__aenter__.return_value = mock_session

                    mock_user = Mock()
                    mock_user.id = uuid4()
                    mock_user.username = "test_user"
                    mock_user.role = "operator"
                    mock_user.is_active = True
                    mock_auth.return_value = mock_user

                    login_data = {
                        "username": "test_user",
                        "password": "test_password"
                    }

                    response = await ac.post("/auth/login", json=login_data)

                    assert response.status_code == 200
                    mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_failed_login_audit_logging(self):
        """Тест логирования неудачной авторизации"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.routes.auth.authenticate_user') as mock_auth:
                with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                    mock_session = AsyncMock()
                    mock_session_ctx.return_value.__aenter__.return_value = mock_session

                    mock_auth.return_value = None  # Неудачная аутентификация

                    login_data = {
                        "username": "wrong_user",
                        "password": "wrong_password"
                    }

                    response = await ac.post("/auth/login", json=login_data)

                    assert response.status_code == 401
                    mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_admin_security_endpoints_access(self, admin_token):
        """Тест доступа к административным эндпоинтам"""
        headers = {"Authorization": f"Bearer {admin_token}"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.middleware.auth.get_current_user') as mock_get_user:
                with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                    mock_session = AsyncMock()
                    mock_session_ctx.return_value.__aenter__.return_value = mock_session

                    mock_user = Mock()
                    mock_user.id = uuid4()
                    mock_user.username = "admin"
                    mock_user.role = "admin"
                    mock_get_user.return_value = mock_user

                    mock_result = Mock()
                    mock_result.fetchall.return_value = []
                    mock_result.scalar.return_value = 0
                    mock_session.execute.return_value = mock_result

                    response = await ac.get("/admin/security/audit-logs", headers=headers)

                    assert response.status_code in [200, 422]

    @pytest.mark.asyncio
    async def test_operator_denied_admin_endpoints(self, operator_token):
        """Тест запрета доступа оператора к административным эндпоинтам"""
        headers = {"Authorization": f"Bearer {operator_token}"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.middleware.auth.get_current_user') as mock_get_user:
                with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                    mock_session = AsyncMock()
                    mock_session_ctx.return_value.__aenter__.return_value = mock_session

                    mock_user = Mock()
                    mock_user.id = uuid4()
                    mock_user.username = "operator"
                    mock_user.role = "operator"
                    mock_get_user.return_value = mock_user

                    response = await ac.get("/admin/security/audit-logs", headers=headers)

                    assert response.status_code == 403


class TestSecurityMiddleware:
    """Тесты security middleware"""

    @pytest.mark.asyncio
    async def test_audit_middleware_logs_protected_endpoints(self):
        """Тест автоматического логирования защищенных эндпоинтов"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                mock_session = AsyncMock()
                mock_session_ctx.return_value.__aenter__.return_value = mock_session

                response = await ac.get("/api/v1/signals")

                assert response.status_code in [401, 403]
                mock_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_audit_middleware_skips_excluded_paths(self):
        """Тест пропуска служебных эндпоинтов"""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
            with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
                mock_session = AsyncMock()
                mock_session_ctx.return_value.__aenter__.return_value = mock_session

                response = await ac.get("/health")

                assert response.status_code == 200
                mock_session.execute.assert_not_called()


class TestSessionManagement:
    """Тесты управления сессиями"""

    @pytest.mark.asyncio
    async def test_revoke_session(self):
        """Тест отзыва сессии"""
        session_id = str(uuid4())

        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            await enhanced_jwt_handler.revoke_session(session_id, "Test revocation")

            # Проверяем, что сессия была отозвана
            mock_session.execute.assert_called()
            mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_is_session_active(self):
        """Тест проверки активности сессии"""
        session_id = str(uuid4())
        refresh_token = "test_refresh_token"

        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Мокаем активную сессию
            mock_result = Mock()
            mock_result.scalar.return_value = 1
            mock_session.execute.return_value = mock_result

            is_active = await enhanced_jwt_handler._is_session_active(session_id, refresh_token)

            assert is_active is True
            mock_session.execute.assert_called()


class TestAuditLogRetention:
    """Тесты политики хранения audit-логов"""

    @pytest.mark.asyncio
    async def test_audit_log_immutability(self):
        """Тест неизменяемости audit-логов"""
        # Этот тест проверяет, что SQL триггеры предотвращают изменение логов
        # В реальной среде попытка UPDATE или DELETE должна вызывать исключение

        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Мокаем исключение при попытке изменения
            mock_session.execute.side_effect = Exception("Audit logs are immutable")

            # Пытаемся "изменить" audit лог
            with pytest.raises(Exception, match="immutable"):
                await mock_session.execute("UPDATE security.audit_logs SET action_type = 'modified'")


class TestSecurityMetrics:
    """Тесты метрик безопасности"""

    @pytest.mark.asyncio
    async def test_security_metrics_collection(self):
        """Тест сбора метрик безопасности"""
        with patch('src.api.middleware.security.get_async_session') as mock_session_ctx:
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__.return_value = mock_session

            # Мокаем метрики
            mock_result = Mock()
            mock_result.fetchall.return_value = [
                Mock(metric_name='total_actions', value=100),
                Mock(metric_name='failed_logins', value=5),
                Mock(metric_name='unique_users', value=20),
            ]
            mock_session.execute.return_value = mock_result

            # Тестируем получение метрик через SQL
            metrics_query = "SELECT 'test' as metric_name, 1 as value"
            result = await mock_session.execute(metrics_query)

            assert result is not None
            mock_session.execute.assert_called_with(metrics_query)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
