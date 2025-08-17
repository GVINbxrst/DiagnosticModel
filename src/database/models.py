# Модели БД (SQLAlchemy) для токовой диагностики

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


class MaintenanceStatus(PyEnum):  # соответствует maintenance_status ENUM в SQL
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TimestampMixin:  # Временные метки
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


class User(Base, TimestampMixin):  # Пользователи
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, values_callable=lambda c: [e.value for e in c], name="user_role", create_type=False),
        nullable=False,
        default=UserRole.VIEWER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Equipment(Base, TimestampMixin):  # Оборудование
    __tablename__ = "equipment"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[EquipmentType] = mapped_column(
        Enum(EquipmentType, values_callable=lambda c: [e.value for e in c], name="equipment_type", create_type=False),
        nullable=False
    )
    status: Mapped[EquipmentStatus] = mapped_column(
        Enum(EquipmentStatus, values_callable=lambda c: [e.value for e in c], name="equipment_status", create_type=False),
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


class DefectType(Base):  # Типы дефектов
    __tablename__ = "defect_types"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    code: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100))
    default_severity: Mapped[DefectSeverity] = mapped_column(
        Enum(DefectSeverity, values_callable=lambda c: [e.value for e in c], name="defect_severity", create_type=False),
        nullable=False,
        default=DefectSeverity.MEDIUM
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )


class ProcessingStatus(PyEnum):  # соответствует processing_status ENUM в SQL
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RawSignal(Base, TimestampMixin):  # Сырые сигналы
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
    # Уникальный SHA256 хеш файла для идемпотентной загрузки.
    # В PostgreSQL будет создан уникальный индекс (см. __table_args__).
    file_hash: Mapped[Optional[str]] = mapped_column(String(64))

    # Статус обработки
    processed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, values_callable=lambda c: [e.value for e in c], name="processing_status", create_type=False),
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
    # Уникальный индекс для предотвращения дубликатов загрузки одного и того же файла.
    Index('uq_raw_signals_file_hash', 'file_hash', unique=True),
    )


class Feature(Base, TimestampMixin):  # Признаки
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

    # Кластеризация (semi-supervised): идентификатор кластера по эмбеддингу
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)

    # Оценка степени развития дефекта (0..1) - вычисляется TCN/аналитикой
    severity_score: Mapped[Optional[float]] = mapped_column(Numeric(precision=5, scale=4))

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


class Prediction(Base, TimestampMixin):  # Прогнозы/аномалии
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

class Forecast(Base, TimestampMixin):  # Прогнозы временных рядов (RMS и др.)
    __tablename__ = "forecasts"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    raw_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("raw_signals.id", ondelete="SET NULL"), index=True)
    equipment_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("equipment.id", ondelete="CASCADE"), index=True)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False, default=24)  # количество шагов прогноза
    method: Mapped[str] = mapped_column(String(50), nullable=False, default="simple_trend")
    forecast_data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    probability_over_threshold: Mapped[Optional[float]] = mapped_column(Numeric(precision=5, scale=4))
    model_version: Mapped[Optional[str]] = mapped_column(String(20))
    # Новое поле для sequence risk (вероятность дефекта в горизонте)
    risk_score: Mapped[Optional[float]] = mapped_column(Numeric(precision=5, scale=4))

    raw_signal = relationship("RawSignal")
    equipment = relationship("Equipment")

    __table_args__ = (
        Index('idx_forecasts_equipment_created', 'equipment_id', 'created_at'),
    )

class AnomalyScore(Base, TimestampMixin):  # Потоковые оценки аномалий
    __tablename__ = "anomaly_scores"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    feature_id: Mapped[UUID] = mapped_column(UniversalUUID(), ForeignKey("features.id", ondelete="CASCADE"), index=True, nullable=False)
    raw_id: Mapped[UUID] = mapped_column(UniversalUUID(), ForeignKey("raw_signals.id", ondelete="CASCADE"), index=True, nullable=False)
    equipment_id: Mapped[UUID] = mapped_column(UniversalUUID(), ForeignKey("equipment.id", ondelete="CASCADE"), index=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)  # stream | stats (legacy значения сохраняются если присутствуют)
    model_version: Mapped[Optional[str]] = mapped_column(String(32))
    score: Mapped[float] = mapped_column(Numeric(precision=12, scale=6), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    threshold: Mapped[Optional[float]] = mapped_column(Numeric(precision=12, scale=6))
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Результаты предсказания / аномалий
    probability: Mapped[float] = mapped_column(Numeric(precision=5, scale=4), nullable=False)
    anomaly_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    confidence: Mapped[float] = mapped_column(Numeric(precision=5, scale=4), nullable=False, default=0.0)
    predicted_severity: Mapped[Optional[DefectSeverity]] = mapped_column(
        Enum(DefectSeverity, values_callable=lambda c: [e.value for e in c], name="defect_severity", create_type=False)
    )
    confidence_score: Mapped[Optional[float]] = mapped_column(Numeric(precision=5, scale=4))
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    prediction_details: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    feature = relationship("Feature", back_populates="predictions")
    defect_type = relationship("DefectType")
    __table_args__ = (
        Index('idx_anomaly_scores_equipment_created', 'equipment_id', 'created_at'),
        Index('idx_anomaly_scores_is_anomaly', 'is_anomaly'),
        Index('idx_predictions_feature_id', 'feature_id'),
        Index('idx_predictions_probability', 'probability'),
        Index('idx_predictions_model', 'model_name', 'model_version'),
    )

class DefectCatalog(Base):  # Справочник дефектов
    __tablename__ = "defect_catalog"

    defect_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    severity_scale: Mapped[Optional[str]] = mapped_column(String(255))  # comma-separated levels
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    cluster_labels = relationship("ClusterLabel", back_populates="defect")


class ClusterLabel(Base):  # Маппинг cluster_id -> defect_id (из справочника)
    __tablename__ = "cluster_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    defect_id: Mapped[str] = mapped_column(String(20), ForeignKey("defect_catalog.defect_id", ondelete="CASCADE"), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    defect = relationship("DefectCatalog", back_populates="cluster_labels")
    __table_args__ = (
        Index('idx_cluster_labels_cluster_id', 'cluster_id'),
        Index('idx_cluster_labels_defect_id', 'defect_id'),
    )

class HourlyFeatureSummary(Base, TimestampMixin):  # Почасовые агрегаты признаков (для ускорения аналитики)
    __tablename__ = "hourly_feature_summary"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[UUID] = mapped_column(
        UniversalUUID(), ForeignKey("equipment.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hour_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    rms_mean: Mapped[Optional[float]] = mapped_column(Numeric(precision=12, scale=6))
    rms_max: Mapped[Optional[float]] = mapped_column(Numeric(precision=12, scale=6))
    rms_min: Mapped[Optional[float]] = mapped_column(Numeric(precision=12, scale=6))
    samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index('uq_hourly_feature_equipment_hour', 'equipment_id', 'hour_start', unique=True),
    )


# Дополнительные модели для логирования и аудита

class SystemLog(Base):  # Системные логи
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


class UserSession(Base):  # Сессии пользователей
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


class MaintenanceEvent(Base, TimestampMixin):  # События обслуживания
    __tablename__ = "maintenance_events"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[UUID] = mapped_column(
        UniversalUUID(),
        ForeignKey("equipment.id", ondelete="CASCADE"),
        nullable=False
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("users.id"))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)  # planned, emergency, inspection, repair
    status: Mapped[MaintenanceStatus] = mapped_column(
        Enum(MaintenanceStatus, values_callable=lambda c: [e.value for e in c], name="maintenance_status", create_type=False),
        nullable=False,
        default=MaintenanceStatus.SCHEDULED
    )
    scheduled_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    description: Mapped[Optional[str]] = mapped_column(Text)
    cost: Mapped[Optional[float]] = mapped_column(Numeric(12, 2))
    parts_replaced: Mapped[Optional[dict]] = mapped_column(JSONB)  # упростим: в SQL массив TEXT[], здесь JSONB для переносимости SQLite
    notes: Mapped[Optional[str]] = mapped_column(Text)

    equipment = relationship("Equipment")
    user = relationship("User")

    __table_args__ = (
        Index('idx_maintenance_equipment_id', 'equipment_id'),
        Index('idx_maintenance_status', 'status'),
        Index('idx_maintenance_scheduled_date', 'scheduled_date'),
    )


class SystemConfig(Base):  # Конфигурация системы
    __tablename__ = "system_config"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index('idx_system_config_key', 'key', unique=True),
    )


class StreamStat(Base):  # Статистика потоковой обработки / дрейфа
    __tablename__ = "stream_stats"

    id: Mapped[UUID] = mapped_column(UniversalUUID(), primary_key=True, default=uuid4)
    equipment_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("equipment.id", ondelete="CASCADE"), index=True)
    feature_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("features.id", ondelete="CASCADE"), index=True)
    raw_id: Mapped[Optional[UUID]] = mapped_column(UniversalUUID(), ForeignKey("raw_signals.id", ondelete="CASCADE"), index=True)
    detector: Mapped[str] = mapped_column(String(50), nullable=False)  # adwin | page_hinkley
    metric: Mapped[str] = mapped_column(String(50), nullable=False)  # например 'rms_a' или 'score'
    value: Mapped[float] = mapped_column(Numeric(precision=14, scale=6), nullable=False)
    drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    __table_args__ = (
        Index('idx_stream_stats_equipment_created', 'equipment_id', 'created_at'),
        Index('idx_stream_stats_detector_metric', 'detector', 'metric'),
    )

# Экспорт требуемых сущностей
__all__ = [
    'Base', 'User', 'Equipment', 'DefectType', 'RawSignal', 'Feature', 'Prediction',
    'SystemLog', 'UserSession', 'ProcessingStatus', 'EquipmentStatus', 'EquipmentType',
    'DefectSeverity', 'UserRole', 'MaintenanceEvent', 'SystemConfig', 'MaintenanceStatus', 'Forecast', 'StreamStat', 'HourlyFeatureSummary'
]
