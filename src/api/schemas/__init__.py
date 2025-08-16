"""
Pydantic схемы для API запросов и ответов
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID
from enum import Enum


class UserRole(str, Enum):
    """Роли пользователей"""
    ADMIN = "admin"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    VIEWER = "viewer"


class ProcessingStatus(str, Enum):
    """Статус обработки данных"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Схемы аутентификации
class LoginRequest(BaseModel):
    """Запрос на авторизацию"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    """Ответ с токенами"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Запрос на обновление токена"""
    refresh_token: str


class UserInfo(BaseModel):
    """Информация о пользователе"""
    id: UUID
    username: str
    email: Optional[str]
    full_name: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime

    # Pydantic v2 style config
    model_config = ConfigDict(from_attributes=True)


# Схемы загрузки файлов
class UploadMetadata(BaseModel):
    """Метаданные для загрузки файла"""
    equipment_id: Optional[UUID] = Field(None, description="ID оборудования")
    sample_rate: Optional[int] = Field(25600, description="Частота дискретизации (Гц)")
    description: Optional[str] = Field(None, description="Описание файла")
    tags: Optional[List[str]] = Field(default_factory=list, description="Теги")


class UploadResponse(BaseModel):
    """Ответ на загрузку файла"""
    success: bool
    message: str
    raw_signal_id: UUID
    equipment_id: UUID
    filename: str
    samples_count: int
    phases_detected: List[str]
    processing_task_id: str
    file_size_bytes: int
    upload_time: datetime


# Схемы сигналов
class PhaseData(BaseModel):
    """Данные одной фазы"""
    phase_name: str = Field(..., description="Название фазы (a, b, c)")
    values: List[float] = Field(..., description="Значения сигнала")
    has_data: bool = Field(..., description="Есть ли данные для фазы")
    samples_count: int = Field(..., description="Количество отсчетов")
    statistics: Optional[Dict[str, float]] = Field(None, description="Базовая статистика")


class SignalResponse(BaseModel):
    """Ответ с данными сигнала"""
    raw_signal_id: UUID
    equipment_id: UUID
    recorded_at: datetime
    sample_rate: int
    total_samples: int
    phases: List[PhaseData]
    metadata: Dict[str, Any]
    processing_status: ProcessingStatus


class SignalListItem(BaseModel):
    """Элемент списка сигналов"""
    raw_signal_id: UUID
    equipment_id: UUID
    recorded_at: datetime
    samples_count: int
    phases_available: List[str]
    processing_status: ProcessingStatus
    file_name: Optional[str]


class SignalListResponse(BaseModel):
    """Список сигналов"""
    signals: List[SignalListItem]
    total_count: int
    page: int
    page_size: int
    has_next: bool


# Схемы аномалий
class AnomalyInfo(BaseModel):
    """Информация об аномалии"""
    id: UUID
    feature_id: UUID
    anomaly_type: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность [0-1]")
    severity: str = Field(..., description="Критичность (low, medium, high, critical)")
    description: str
    detected_at: datetime
    window_start: datetime
    window_end: datetime
    affected_phases: List[str]
    model_name: str
    model_version: str
    prediction_data: Dict[str, Any]


class ForecastInfo(BaseModel):
    """Информация о прогнозе"""
    equipment_id: UUID
    forecast_horizon_hours: int
    max_anomaly_probability: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    generated_at: datetime
    phases_analyzed: List[str]
    forecast_details: Dict[str, Any]


class AnomaliesResponse(BaseModel):
    """Ответ со списком аномалий"""
    equipment_id: UUID
    anomalies: List[AnomalyInfo]
    forecast: Optional[ForecastInfo]
    total_anomalies: int
    period_start: datetime
    period_end: datetime
    summary: Dict[str, Any]


# Схемы оборудования
class EquipmentInfo(BaseModel):
    """Информация об оборудовании"""
    id: UUID
    name: str
    equipment_type: str
    model: Optional[str]
    location: Optional[str]
    is_active: bool
    installed_at: Optional[datetime]
    last_maintenance: Optional[datetime]
    health_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class EquipmentListResponse(BaseModel):
    """Список оборудования"""
    equipment: List[EquipmentInfo]
    total_count: int
    active_count: int


# Схемы фильтрации и пагинации
class TimeRangeFilter(BaseModel):
    """Фильтр по времени"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(1, ge=1, description="Номер страницы")
    page_size: int = Field(20, ge=1, le=100, description="Размер страницы")


class AnomalyFilter(BaseModel):
    """Фильтр аномалий"""
    time_range: Optional[TimeRangeFilter] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    severity_levels: Optional[List[str]] = None
    phases: Optional[List[str]] = None
    pagination: PaginationParams = PaginationParams()


# Схемы мониторинга
class HealthResponse(BaseModel):
    """Ответ о состоянии здоровья системы"""
    status: str = Field(..., description="healthy или unhealthy")
    message: str
    version: str
    timestamp: float
    checks: Optional[Dict[str, Any]] = None


class SystemStats(BaseModel):
    """Системная статистика"""
    total_signals: int
    processed_signals: int
    failed_signals: int
    total_anomalies: int
    active_equipment: int
    processing_queue_size: int
    last_update: datetime


class MetricsResponse(BaseModel):
    """Ответ с метриками"""
    metrics: Dict[str, Any]
    generated_at: datetime


# Схемы ошибок
class ErrorResponse(BaseModel):
    """Стандартный ответ об ошибке"""
    error: bool = True
    status_code: int
    message: str
    detail: Optional[str] = None
    timestamp: float


class ValidationErrorDetail(BaseModel):
    """Детали ош��бки валидации"""
    field: str
    message: str
    invalid_value: Any


class ValidationErrorResponse(BaseModel):
    """Ответ с ошибками валидации"""
    error: bool = True
    status_code: int = 422
    message: str = "Ошибка валидации данных"
    validation_errors: List[ValidationErrorDetail]
    timestamp: float


# Схемы для задач
class TaskInfo(BaseModel):
    """Информация о фоновой задаче"""
    task_id: str
    task_name: str
    status: str
    progress: Optional[int] = Field(None, ge=0, le=100)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class TaskStatusResponse(BaseModel):
    """Ответ со статусом задачи"""
    task: TaskInfo
    estimated_completion: Optional[datetime] = None


# Валидаторы
class UploadMetadata(UploadMetadata):
    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v is not None and (v < 1000 or v > 100000):
            raise ValueError('Частота дискретизации должна быть от 1000 до 100000 Гц')
        return v

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if v and len(v) > 10:
            raise ValueError('Максимум 10 тегов')
        return v


class AnomalyFilter(AnomalyFilter):
    @field_validator('time_range')
    @classmethod
    def validate_time_range(cls, v):
        if v and v.start_date and v.end_date and v.start_date >= v.end_date:
            raise ValueError('Начальная дата должна быть раньше конечной')
        return v
