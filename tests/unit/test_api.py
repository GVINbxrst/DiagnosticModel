"""
Тесты для FastAPI приложения системы диагностики двигателей
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
import json
import io

from src.api.main import app
from src.api.middleware.auth import jwt_handler
from src.database.models import User, Equipment, RawSignal


@pytest.fixture
def client():
    """HTTP клиент для тестирования"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Асинхронный HTTP клиент"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
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


class TestAuthEndpoints:
    """Тесты эндпоинтов авторизации"""

    @patch('src.api.routes.auth.authenticate_user')
    def test_login_success(self, mock_auth, client, mock_user):
        """Тест успешной авторизации"""
        mock_auth.return_value = mock_user

        login_data = {
            "username": "test_user",
            "password": "test_password"
        }

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @patch('src.api.routes.auth.authenticate_user')
    def test_login_invalid_credentials(self, mock_auth, client):
        """Тест авторизации с неверными данными"""
        mock_auth.return_value = None

        login_data = {
            "username": "wrong_user",
            "password": "wrong_password"
        }

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 401

        data = response.json()
        assert data["detail"] == "Неверное имя пользователя или пароль"

    def test_protected_endpoint_without_token(self, client):
        """Тест защищенного эндпоинта без токена"""
        response = client.get("/api/v1/signals")
        assert response.status_code == 403  # Или 401 в зависимости от реализации


class TestUploadEndpoints:
    """Тесты эндпоинтов загрузки файлов"""

    @patch('src.api.routes.upload.CSVLoader')
    @patch('src.api.routes.upload.process_raw')
    def test_upload_csv_success(self, mock_process_raw, mock_csv_loader, client, auth_headers, mock_equipment):
        """Тест успешной загрузки CSV файла"""
        # Мокаем CSV загрузчик
        mock_loader_instance = Mock()
        mock_stats = Mock()
        mock_stats.raw_signal_id = uuid4()
        mock_stats.total_samples = 1000
        mock_stats.phase_a_samples = 1000
        mock_stats.phase_b_samples = 1000
        mock_stats.phase_c_samples = 0
        mock_stats.upload_time = "2024-01-01T00:00:00"

        mock_loader_instance.load_csv_file.return_value = mock_stats
        mock_csv_loader.return_value = mock_loader_instance

        # Мокаем задачу worker
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_process_raw.delay.return_value = mock_task

        # Мокаем создание оборудования
        with patch('src.api.routes.upload._create_equipment_from_filename') as mock_create_eq:
            mock_create_eq.return_value = mock_equipment.id

            # Создаем тестовый CSV файл
            csv_content = "current_R,current_S,current_T\n1.0,2.0,\n3.0,4.0,"
            files = {"file": ("test.csv", io.StringIO(csv_content), "text/csv")}

            response = client.post(
                "/api/v1/upload",
                files=files,
                headers=auth_headers
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "raw_signal_id" in data
        assert "processing_task_id" in data

    def test_upload_non_csv_file(self, client, auth_headers):
        """Тест загрузки не-CSV файла"""
        files = {"file": ("test.txt", io.StringIO("not a csv"), "text/plain")}

        response = client.post(
            "/api/v1/upload",
            files=files,
            headers=auth_headers
        )

        assert response.status_code == 400
        data = response.json()
        assert "CSV" in data["detail"]


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


class TestSignalsEndpoints:
    """Тесты эндпоинтов сигналов"""

    @patch('src.api.routes.signals.get_async_session')
    def test_get_signal_data_success(self, mock_session, client, auth_headers):
        """Тест получения данных сигнала"""
        # Мокаем сигнал
        mock_signal = Mock(spec=RawSignal)
        mock_signal.id = uuid4()
        mock_signal.equipment_id = uuid4()
        mock_signal.sample_rate = 25600
        mock_signal.samples_count = 1000
        mock_signal.metadata = {}
        mock_signal.processing_status = "completed"

        # Мокаем сжатые данные фаз
    import numpy as np
    from src.utils.serialization import dump_float32_array
    test_data = np.random.normal(0, 1, 100).astype(np.float32)
    compressed_data = dump_float32_array(test_data)

        mock_signal.phase_a = compressed_data
        mock_signal.phase_b = compressed_data
        mock_signal.phase_c = None

        # Мокаем сессию БД
        mock_db_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db_session

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_signal
        mock_db_session.execute.return_value = mock_result

        response = client.get(
            f"/api/v1/signals/{mock_signal.id}",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["raw_signal_id"] == str(mock_signal.id)
        assert "phases" in data
        assert len(data["phases"]) == 3

    @patch('src.api.routes.signals.get_async_session')
    def test_get_signal_data_not_found(self, mock_session, client, auth_headers):
        """Тест получения несуществующего сигнала"""
        mock_db_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db_session

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        signal_id = uuid4()
        response = client.get(
            f"/api/v1/signals/{signal_id}",
            headers=auth_headers
        )

        assert response.status_code == 404


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
            "exp": datetime.utcnow() - timedelta(hours=1),  # Истек час назад
            "iat": datetime.utcnow() - timedelta(hours=2)
        }

        expired_token = jwt.encode(expired_payload, "secret", algorithm="HS256")
        headers = {"Authorization": f"Bearer {expired_token}"}

        response = client.get("/api/v1/signals", headers=headers)
        assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
