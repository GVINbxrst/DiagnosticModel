"""
SQLAlchemy модели для DiagMod
Модели базы данных для токовой диагностики двигателей
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, DateTime, Enum, Integer, LargeBinary, Numeric, String, Text,
    func, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base

# Компиляция JSONB для SQLite (тестовый режим) -> JSON (текст)
@compiles(JSONB, 'sqlite')
def compile_jsonb_sqlite(type_, compiler, **kw):  # pragma: no cover - инфраструктурный слой
    return 'JSON'

# Универсальный UUID тип для Postgres/SQLite: хранит UUID как native UUID в PG и как текст (36) в SQLite.
class UniversalUUID(TypeDecorator):  # pragma: no cover - инфраструктурный слой
    impl = CHAR(36)
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, UUID):
            return value if dialect.name == 'postgresql' else str(value)
        if isinstance(value, str):
            from uuid import UUID as _UUID
            try:
                u = _UUID(value)
            except Exception as e:  # pragma: no cover
                raise TypeError(f"Invalid UUID string '{value}': {e}") from e
            return u if dialect.name == 'postgresql' else str(u)
        raise TypeError(f"Unsupported UUID value type: {type(value)}")

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, UUID):
            return value
        from uuid import UUID as _UUID
        try:
            return _UUID(str(value))
        except Exception as e:  # pragma: no cover
            raise TypeError(f"Invalid UUID value from DB '{value}': {e}") from e

# Базовый класс для всех моделей
Base = declarative_base()

# Перечисления (соответствуют SQL ENUM типам)
from enum import Enum as PyEnum

class EquipmentStatus(PyEnum):
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    INACTIVE = "inactive"
    FAULT = "fault"

class EquipmentType(PyEnum):
    INDUCTION_MOTOR = "induction_motor"
    SYNCHRONOUS_MOTOR = "synchronous_motor"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    FAN = "fan"
    CONVEYOR = "conveyor"

class DefectSeverity(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UserRole(PyEnum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    OPERATOR = "operator"
    VIEWER = "viewer"


class TimestampMixin:
    """Миксин для добавления временных меток"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class User(Base, TimestampMixin):
    """Пользователи системы"""
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, values_callable=lambda c: [e.value for e in c]), nullable=False, default=UserRole.VIEWER)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Equipment(Base, TimestampMixin):
    """Оборудование (двигатели, насосы и т.д.)"""
    __tablename__ = "equipment"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[EquipmentType] = mapped_column(Enum(EquipmentType, values_callable=lambda c: [e.value for e in c]), nullable=False)
    status: Mapped[EquipmentStatus] = mapped_column(
        Enum(EquipmentStatus, values_callable=lambda c: [e.value for e in c]),
        nullable=False,
        default=EquipmentStatus.INACTIVE
    )
    manufacturer: Mapped[Optional[str]] = mapped_column(String(255))
    model: Mapped[Optional[str]] = mapped_column(String(255))
    serial_number: Mapped[Optional[str]] = mapped_column(String(255))
    installation_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    location: Mapped[Optional[str]] = mapped_column(String(500))
    specifications: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Связи
    raw_signals = relationship("RawSignal", back_populates="equipment")


class DefectType(Base):
    """Типы дефектов"""
    __tablename__ = "defect_types"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100))
    default_severity: Mapped[DefectSeverity] = mapped_column(Enum(DefectSeverity, values_callable=lambda c: [e.value for e in c]),
        nullable=False,
        default=DefectSeverity.MEDIUM
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class ProcessingStatus(PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RawSignal(Base, TimestampMixin):
    """Сырые токовые сигналы"""
    __tablename__ = "raw_signals"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("equipment.id", ondelete="CASCADE"),
        nullable=False
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sample_rate_hz: Mapped[int] = mapped_column(Integer, nullable=False)
    samples_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Данные фаз в сжатом виде (gzip + float32)
    phase_a: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # Фаза R
    phase_b: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # Фаза S
    phase_c: Mapped[Optional[bytes]] = mapped_column(LargeBinary)  # Фаза T

    # Метаданные
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Информация о файле
    file_name: Mapped[Optional[str]] = mapped_column(String(500))
    file_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Статус обработки
    processed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, values_callable=lambda c: [e.value for e in c]),
        nullable=False,
        default=ProcessingStatus.PENDING,
        index=True
    )

    # Связи
    equipment = relationship("Equipment", back_populates="raw_signals")
    features = relationship("Feature", back_populates="raw_signal")

    # Индексы
    __table_args__ = (
        Index('idx_raw_signals_recorded_at', 'recorded_at'),
        Index('idx_raw_signals_equipment_time', 'equipment_id', 'recorded_at'),
        Index('idx_raw_signals_unprocessed', 'processed', 'created_at'),
        Index('idx_raw_signals_file_hash', 'file_hash'),
    )


class Feature(Base, TimestampMixin):
    """Извлеченные признаки"""
    __tablename__ = "features"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    raw_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("raw_signals.id", ondelete="CASCADE"),
        nullable=False
    )

    # Временное окно
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Статистические признаки для каждой фазы
    # RMS (Root Mean Square)
    rms_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    rms_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    rms_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    # Crest Factor
    crest_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    crest_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    crest_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    # Kurtosis
    kurt_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    kurt_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    kurt_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    # Skewness
    skew_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    skew_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    skew_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    # Базовая статистика
    mean_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    mean_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    mean_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    std_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    std_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    std_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    min_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    min_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    min_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    max_a: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    max_b: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))
    max_c: Mapped[Optional[float]] = mapped_column(Numeric(precision=10, scale=6))

    # Частотные характеристики
    fft_spectrum: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Дополнительные признаки
    extra: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Связи
    raw_signal = relationship("RawSignal", back_populates="features")
    predictions = relationship("Prediction", back_populates="feature")
    # Индексы
    __table_args__ = (
        Index('idx_features_raw_id', 'raw_id'),
        Index('idx_features_window_start', 'window_start'),
        Index('idx_features_window_range', 'window_start', 'window_end'),
    )

## ProcessingStatus уже определён на верхнем уровне


class Prediction(Base, TimestampMixin):
    """Прогнозы и детекция аномалий"""
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    feature_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("features.id", ondelete="CASCADE"),
        nullable=False
    )
    equipment_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), nullable=True, index=True)
    defect_type_id: Mapped[Optional[UUID]] = mapped_column(
        UniversalUUID(),
        ForeignKey("defect_types.id")
    )

    # Результаты предсказания
    probability: Mapped[float] = mapped_column(Numeric(precision=5, scale=4), nullable=False)
    # Флаг обнаружения аномалии (для запросов аномалий)
    anomaly_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    # Уверенность (дублирующая метрика для удобства фильтрации)
    confidence: Mapped[float] = mapped_column(Numeric(precision=5, scale=4), nullable=False, default=0.0)
    predicted_severity: Mapped[Optional[DefectSeverity]] = mapped_column(
        Enum(DefectSeverity, values_callable=lambda c: [e.value for e in c])
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(Numeric(precision=5, scale=4))

    # Информация о модели
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    model_type: Mapped[Optional[str]] = mapped_column(String(50))

    # Дополнительные результаты
    prediction_details: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Связи
    feature = relationship("Feature", back_populates="predictions")
    defect_type = relationship("DefectType")

    # Индексы
    __table_args__ = (
        Index('idx_predictions_feature_id', 'feature_id'),
        Index('idx_predictions_probability', 'probability'),
        Index('idx_predictions_model', 'model_name', 'model_version'),
    )


# Дополнительные модели для логирования и аудита

class SystemLog(Base):
    """Системные логи"""
    __tablename__ = "system_logs"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    level: Mapped[str] = mapped_column(String(20), nullable=False)
    module: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    user_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("users.id"))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 support
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Связи
    user = relationship("User")

    # Индексы
    __table_args__ = (
        Index('idx_system_logs_created_at', 'created_at'),
        Index('idx_system_logs_level', 'level', 'created_at'),
        Index('idx_system_logs_module', 'module', 'created_at'),
    )


class UserSession(Base):
    """Пользовательские сессии"""
    __tablename__ = "user_sessions"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Связи
    user = relationship("User")

    # Индексы
    __table_args__ = (
        Index('idx_user_sessions_user_id', 'user_id'),
        Index('idx_user_sessions_token_hash', 'token_hash'),
        Index('idx_user_sessions_expires_at', 'expires_at'),
    )

# Экспорт требуемых сущностей
__all__ = [
    'Base', 'User', 'Equipment', 'DefectType', 'RawSignal', 'Feature', 'Prediction',
    'SystemLog', 'UserSession', 'ProcessingStatus', 'EquipmentStatus', 'EquipmentType',
    'DefectSeverity', 'UserRole'
]
