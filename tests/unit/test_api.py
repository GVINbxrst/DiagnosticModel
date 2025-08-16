"""
Тесты для FastAPI приложения системы диагностики двигателей
"""

import pytest
import asyncio
import httpx
from httpx import ASGITransport
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import json
import io
from datetime import datetime, timedelta, UTC

from src.api.main import app
from src.api.middleware.auth import jwt_handler
from src.database.models import User, Equipment, RawSignal


@pytest.fixture
def client():
    """Синхронная обёртка над AsyncClient для совместимости старых тестов."""
    transport = ASGITransport(app=app)
    async_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    class SyncBridge:
        def __init__(self, ac):
            self._ac = ac
            self.app = app
        def get(self, *a, **kw):
            return asyncio.run(self._ac.get(*a, **kw))
        def post(self, *a, **kw):
            return asyncio.run(self._ac.post(*a, **kw))
        def put(self, *a, **kw):
            return asyncio.run(self._ac.put(*a, **kw))
        def delete(self, *a, **kw):
            return asyncio.run(self._ac.delete(*a, **kw))
    client_obj = SyncBridge(async_client)
    return client_obj


@pytest.fixture
async def async_client():
    """Асинхронный HTTP клиент с явным ASGITransport"""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.fixture
def mock_user():
    """Мок пользователя для тестов"""
    user = Mock(spec=User)
    user.id = uuid4()
    user.username = "test_user"
    user.email = "test@example.com"
    user.role = "engineer"
    user.is_active = True
    user.password_hash = jwt_handler.hash_password("test_password")
    return user


@pytest.fixture
def auth_headers(mock_user):
    """Заголовки с JWT токеном для авторизованных запросов"""
    access_token = jwt_handler.create_access_token(
        user_id=str(mock_user.id),
        username=mock_user.username,
        role=mock_user.role
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def mock_equipment():
    """Мок оборудования"""
    equipment = Mock(spec=Equipment)
    equipment.id = uuid4()
    equipment.name = "Test Motor 001"
    equipment.equipment_type = "motor"
    equipment.is_active = True
    return equipment


class TestHealthEndpoints:
    """Тесты эндпоинтов здоровья"""

    def test_root_endpoint(self, client):
        """Тест корневого эндпоинта"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["message"] == "DiagMod API работает корректно"
        assert data["version"] == "1.0.0"

    @patch('src.api.main.get_async_session')
    def test_health_check_success(self, mock_session, client):
        """Тест успешной проверки здоровья"""
        # Мокаем успешное подключение к БД
        mock_session.return_value.__aenter__ = AsyncMock()
        mock_session.return_value.__aexit__ = AsyncMock()

        mock_db_session = AsyncMock()
        mock_db_session.execute = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db_session

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "checks" in data
        assert "database" in data["checks"]

    @patch('src.api.main.get_async_session')
    def test_health_check_db_failure(self, mock_session, client):
        """Тест проверки здоровья при проблемах с БД"""
        # Мокаем ошибку подключения к БД
        mock_session.side_effect = Exception("Database connection failed")

        response = client.get("/health")
        assert response.status_code == 503

        data = response.json()
        assert data["status"] == "unhealthy"


class TestSignalsEndpoints:
    """Тесты эндпоинтов сигналов"""
    def test_get_signal_data_success(self, client, auth_headers):
        """Создаём реальный RawSignal в тестовой SQLite и запрашиваем через API."""
        from src.api.middleware import auth as auth_module
        from src.database.connection import get_async_session
        from src.database.models import ProcessingStatus
        from src.utils.serialization import dump_float32_array
        import numpy as np

        client.app.dependency_overrides[auth_module.require_any_role] = lambda: Mock(username="tester", id=uuid4())

        raw_id = uuid4()
        equipment_id = uuid4()
        test_data = np.random.normal(0, 1, 64).astype(np.float32)
        compressed = dump_float32_array(test_data)

        async def seed():
            async with get_async_session() as s:  # ensures schema
                signal = RawSignal(
                    id=raw_id,
                    equipment_id=equipment_id,
                    sample_rate_hz=25600,
                    samples_count=1000,
                    # Поле модели называется meta, не metadata
                    meta={"original_filename": "test.csv"},
                    processing_status=ProcessingStatus.COMPLETED,
                    recorded_at=datetime.now(UTC),
                    file_hash="hash123",
                    phase_a=compressed,
                    phase_b=compressed,
                    phase_c=None
                )
                s.add(signal)
                await s.commit()

        asyncio.run(seed())

        response = client.get(f"/api/v1/signals/{raw_id}", headers=auth_headers)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["raw_signal_id"] == str(raw_id)
        assert len(body["phases"]) == 3

    def test_get_signal_data_not_found(self, client, auth_headers):
        from src.api.middleware import auth as auth_module
        from src.database.connection import db_session, get_async_session

        client.app.dependency_overrides[auth_module.require_any_role] = lambda: Mock(username="tester", id=uuid4())
        mock_db_session = AsyncMock()

        async def override_session():
            yield mock_db_session

        client.app.dependency_overrides[db_session] = override_session
        client.app.dependency_overrides[get_async_session] = override_session

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        signal_id = uuid4()
        response = client.get(f"/api/v1/signals/{signal_id}", headers=auth_headers)
        assert response.status_code == 404


class TestAnomaliesEndpoints:
    """Тесты эндпоинтов аномалий"""

    @patch('src.api.routes.anomalies.get_async_session')
    def test_get_equipment_anomalies_success(self, mock_session, client, auth_headers, mock_equipment):
        """Тест получения аномалий для оборудования"""
        # Мокаем сессию БД
        mock_db_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db_session

        # Мокаем результаты запросов
        mock_equipment_result = Mock()
        mock_equipment_result.scalar_one_or_none.return_value = mock_equipment

        mock_anomalies_result = Mock()
        mock_anomalies_result.scalars.return_value.all.return_value = []

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 0

        mock_db_session.execute.side_effect = [
            mock_equipment_result,  # Запрос оборудования
            mock_count_result,      # Подсчет аномалий
            mock_anomalies_result   # Запрос аномалий
        ]

        response = client.get(
            f"/api/v1/anomalies/{mock_equipment.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["equipment_id"] == str(mock_equipment.id)
        assert "anomalies" in data
        assert "total_anomalies" in data

    @patch('src.api.routes.anomalies.get_async_session')
    def test_get_equipment_anomalies_not_found(self, mock_session, client, auth_headers):
        """Тест получения аномалий для несуществующего оборудования"""
        mock_db_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db_session

        # Мокаем отсутствие оборудования
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        equipment_id = uuid4()
        response = client.get(
            f"/api/v1/anomalies/{equipment_id}",
            headers=auth_headers
        )

        assert response.status_code == 404
        data = response.json()
        assert "не найдено" in data["detail"]




class TestMonitoringEndpoints:
    """Тесты эндпоинтов мониторинга"""

    def test_metrics_endpoint(self, client):
        """Тест эндпоинта метрик Prometheus"""
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]

    def test_monitoring_health(self, client):
        """Тест эндпоинта здоровья мониторинга"""
        response = client.get("/monitoring/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestErrorHandling:
    """Тесты обработки ошибок"""

    def test_404_error(self, client):
        """Тест 404 ошибки"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_validation_error(self, client, auth_headers):
        """Тест ошибки валидации"""
        # Отправляем невалидные данные
        invalid_data = {"invalid_field": "value"}
        response = client.post("/auth/login", json=invalid_data)
        assert response.status_code == 422


class TestSecurity:
    """Тесты безопасности"""

    def test_unauthorized_access(self, client):
        """Тест неавторизованного доступа к защищенным эндпоинтам"""
        protected_endpoints = [
            "/api/v1/upload",
            "/api/v1/signals",
            "/api/v1/anomalies/123e4567-e89b-12d3-a456-426614174000"
        ]

        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code in [401, 403]

    def test_invalid_token(self, client):
        """Тест с невалидным токеном"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/signals", headers=headers)
        assert response.status_code == 401

    def test_expired_token(self, client):
        """Тест с истекшим токеном"""
        # Создаем токен с истекшим временем
        import jwt
        from datetime import datetime, timedelta

        expired_payload = {
            "sub": str(uuid4()),
            "username": "test",
            "role": "engineer",
            "type": "access",
            "exp": datetime.now(UTC) - timedelta(hours=1),  # Истек час назад
            "iat": datetime.now(UTC) - timedelta(hours=2)
        }

        expired_token = jwt.encode(expired_payload, "secret", algorithm="HS256")
        headers = {"Authorization": f"Bearer {expired_token}"}

        response = client.get("/api/v1/signals", headers=headers)
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
