"""
Настройки приложения DiagMod
Конфигурация на основе переменных окружения
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки приложения"""

    # Основные настройки приложения
    APP_NAME: str = "DiagMod"
    APP_VERSION: str = "1.0.0"
    APP_ENVIRONMENT: str = Field(default="development", env="APP_ENVIRONMENT")
    APP_DEBUG: bool = Field(default=False, env="APP_DEBUG")

    # Настройки базы данных
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://diagmod_user:diagmod_password@localhost:5432/diagmod",
        env="DATABASE_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")

    # Настройки Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Настройки Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")

    # Настройки API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=4, env="API_WORKERS")
    API_RELOAD: bool = Field(default=False, env="API_RELOAD")

    # Настройки безопасности
    SECRET_KEY: str = Field(default="your-super-secret-key", env="SECRET_KEY")
    JWT_SECRET_KEY: str = Field(default="your-jwt-secret-key", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")

    # Настройки CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )

    # Настройки логирования
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    LOG_FILE_PATH: Optional[str] = Field(default="./logs/app.log", env="LOG_FILE_PATH")

    # Настройки обработки данных
    DATA_PATH: str = Field(default="./data", env="DATA_PATH")
    CSV_BATCH_SIZE: int = Field(default=10000, env="CSV_BATCH_SIZE")
    FEATURE_EXTRACTION_WORKERS: int = Field(default=4, env="FEATURE_EXTRACTION_WORKERS")

    # Настройки ML моделей
    MODELS_PATH: str = Field(default="./models", env="MODELS_PATH")
    MODEL_AUTO_RETRAIN: bool = Field(default=True, env="MODEL_AUTO_RETRAIN")
    ANOMALY_THRESHOLD: float = Field(default=0.95, env="ANOMALY_THRESHOLD")

    # Настройки сигналов
    SIGNAL_SAMPLING_RATE: int = Field(default=25600, env="SIGNAL_SAMPLING_RATE")
    FFT_WINDOW_SIZE: int = Field(default=1024, env="FFT_WINDOW_SIZE")
    OVERLAP_RATIO: float = Field(default=0.5, env="OVERLAP_RATIO")

    # Настройки мониторинга
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    PROMETHEUS_PORT: int = Field(default=8001, env="PROMETHEUS_PORT")

    # Настройки Dashboard
    DASHBOARD_HOST: str = Field(default="0.0.0.0", env="DASHBOARD_HOST")
    DASHBOARD_PORT: int = Field(default=8501, env="DASHBOARD_PORT")
    API_URL: str = Field(default="http://localhost:8000", env="API_URL")

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @validator("APP_ENVIRONMENT")
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production", "test"]
        if v.lower() not in valid_envs:
            raise ValueError(f"APP_ENVIRONMENT must be one of {valid_envs}")
        return v.lower()

    @property
    def is_development(self) -> bool:
        return self.APP_ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        return self.APP_ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.APP_ENVIRONMENT == "test"

    @property
    def data_path(self) -> Path:
        return Path(self.DATA_PATH)

    @property
    def models_path(self) -> Path:
        return Path(self.MODELS_PATH)

    @property
    def log_file_path(self) -> Optional[Path]:
        return Path(self.LOG_FILE_PATH) if self.LOG_FILE_PATH else None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Получить настройки приложения (с кешированием)"""
    return Settings()
