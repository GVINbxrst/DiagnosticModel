-- =============================================================================
-- DiagMod Database Schema - Initial Setup
-- Создание основной схемы базы данных для диагностики двигателей
-- =============================================================================

-- Создание расширений PostgreSQL
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Создание схемы приложения
CREATE SCHEMA IF NOT EXISTS diagmod;
SET search_path TO diagmod, public;

-- =============================================================================
-- ПЕРЕЧИСЛЕНИЯ (ENUMS)
-- =============================================================================

-- Статусы оборудования
CREATE TYPE equipment_status AS ENUM (
    'active',      -- Активное
    'maintenance', -- На обслуживании
    'inactive',    -- Неактивное
    'fault'        -- Неисправное
);

-- Типы оборудования
CREATE TYPE equipment_type AS ENUM (
    'induction_motor',    -- Асинхронный двигатель
    'synchronous_motor',  -- Синхронный двигатель
    'pump',              -- Насос
    'compressor',        -- Компрессор
    'fan',               -- Вентилятор
    'conveyor'           -- Конвейер
);

-- Критичность дефектов
CREATE TYPE defect_severity AS ENUM (
    'low',       -- Низкая
    'medium',    -- Средняя
    'high',      -- Высокая
    'critical'   -- Критическая
);

-- Статусы обработки сырых сигналов (pipeline)
CREATE TYPE processing_status AS ENUM (
    'pending',      -- ожидает обработки
    'processing',   -- в процессе обработки
    'completed',    -- успешно обработан
    'failed'        -- обработка завершилась ошибкой
);

-- Статусы обслуживания
CREATE TYPE maintenance_status AS ENUM (
    'scheduled',   -- Запланировано
    'in_progress', -- В процессе
    'completed',   -- Завершено
    'cancelled'    -- Отменено
);

-- Роли пользователей
CREATE TYPE user_role AS ENUM (
    'admin',       -- Администратор
    'engineer',    -- Инженер
    'operator',    -- Оператор
    'viewer'       -- Наблюдатель
);

-- =============================================================================
-- ФУНКЦИИ ДЛЯ РАБОТЫ С ВРЕМЕННЫМИ МЕТКАМИ
-- =============================================================================

-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Функция для генерации equipment_id
CREATE OR REPLACE FUNCTION generate_equipment_id()
RETURNS TEXT AS $$
BEGIN
    RETURN 'EQ_' || EXTRACT(YEAR FROM NOW()) || '_' ||
           LPAD(nextval('equipment_id_seq')::TEXT, 6, '0');
END;
$$ language 'plpgsql';

-- Последовательность для equipment_id
CREATE SEQUENCE IF NOT EXISTS equipment_id_seq START 1;

-- =============================================================================
-- ОСНОВНЫЕ ТАБЛИЦЫ
-- =============================================================================

-- Пользователи системы
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'viewer',
    is_active BOOLEAN NOT NULL DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Оборудование
CREATE TABLE equipment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id VARCHAR(50) UNIQUE NOT NULL DEFAULT generate_equipment_id(),
    name VARCHAR(255) NOT NULL,
    type equipment_type NOT NULL,
    status equipment_status NOT NULL DEFAULT 'inactive',
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    serial_number VARCHAR(255),
    installation_date DATE,
    location VARCHAR(500),
    specifications JSONB, -- Технические характеристики
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Типы дефектов
CREATE TABLE defect_types (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100), -- bearing, rotor, stator, electrical, mechanical
    default_severity defect_severity NOT NULL DEFAULT 'medium',
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- События обслуживания
CREATE TABLE maintenance_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(100) NOT NULL, -- planned, emergency, inspection, repair
    status maintenance_status NOT NULL DEFAULT 'scheduled',
    scheduled_date TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    description TEXT,
    cost DECIMAL(12,2),
    parts_replaced TEXT[],
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ТАБЛИЦЫ ДАННЫХ И АНАЛИЗА
-- =============================================================================

-- Сырые сигналы (основная таблица данных)
CREATE TABLE raw_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL,
    sample_rate_hz INTEGER NOT NULL CHECK (sample_rate_hz > 0),
    samples_count INTEGER NOT NULL CHECK (samples_count > 0),
    duration_ms INTEGER GENERATED ALWAYS AS ((samples_count * 1000) / sample_rate_hz) STORED,

    -- Данные фаз в бинарном формате (сжатые float32 массивы)
    phase_a BYTEA, -- Фаза R
    phase_b BYTEA, -- Фаза S (может быть NULL)
    phase_c BYTEA, -- Фаза T (может быть NULL)

    -- Метаданные записи
    meta JSONB DEFAULT '{}', -- Дополнительная информация о записи

    -- Служебные поля
    file_name VARCHAR(500), -- Имя исходного CSV файла
    file_hash VARCHAR(64),  -- SHA256 хеш файла для дедупликации
    processed BOOLEAN NOT NULL DEFAULT false,
    processing_status processing_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Проверки целостности
    CONSTRAINT valid_phases CHECK (
        phase_a IS NOT NULL OR phase_b IS NOT NULL OR phase_c IS NOT NULL
    )
);

-- Извлеченные признаки
CREATE TABLE features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_id UUID NOT NULL REFERENCES raw_signals(id) ON DELETE CASCADE,

    -- Временное окно анализа
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    window_size_ms INTEGER GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (window_end - window_start)) * 1000
    ) STORED,

    -- Статистические признаки для каждой фазы
    -- RMS (Root Mean Square)
    rms_a REAL,
    rms_b REAL,
    rms_c REAL,

    -- Crest Factor (пик-фактор)
    crest_a REAL,
    crest_b REAL,
    crest_c REAL,

    -- Kurtosis (эксцесс)
    kurt_a REAL,
    kurt_b REAL,
    kurt_c REAL,

    -- Skewness (асимметрия)
    skew_a REAL,
    skew_b REAL,
    skew_c REAL,

    -- Дополнительные статистические признаки
    mean_a REAL,
    mean_b REAL,
    mean_c REAL,

    std_a REAL,
    std_b REAL,
    std_c REAL,

    min_a REAL,
    min_b REAL,
    min_c REAL,

    max_a REAL,
    max_b REAL,
    max_c REAL,

    -- Частотные характеристики
    fft_spectrum JSONB, -- Спектр FFT и характеристические частоты

    -- Дополнительные признаки
    extra JSONB DEFAULT '{}', -- Дополнительные вычисленные признаки

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Проверки
    CONSTRAINT valid_window CHECK (window_end > window_start),
    CONSTRAINT valid_rms CHECK (
        (rms_a IS NULL OR rms_a >= 0) AND
        (rms_b IS NULL OR rms_b >= 0) AND
        (rms_c IS NULL OR rms_c >= 0)
    )
);

-- Прогнозы и детекция аномалий
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id UUID NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    equipment_id UUID REFERENCES equipment(id),
    defect_type_id UUID REFERENCES defect_types(id),

    -- Результаты предсказания
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    anomaly_detected BOOLEAN NOT NULL DEFAULT false,
    confidence REAL NOT NULL DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    predicted_severity defect_severity,
    confidence_score REAL CHECK (confidence_score >= 0 AND confidence_score <= 1),

    -- Информация о модели
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50), -- anomaly_detection, classification, regression

    -- Дополнительные результаты
    prediction_details JSONB DEFAULT '{}', -- Детали предсказания

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ВСПОМОГАТЕЛЬНЫЕ ТАБЛИЦЫ
-- =============================================================================

-- Сессии пользователей
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Логи системы
CREATE TABLE system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    module VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    user_id UUID REFERENCES users(id),
    ip_address INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Конфигурация системы
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ТРИГГЕРЫ ДЛЯ АВТОМАТИЧЕСКОГО ОБНОВЛЕНИЯ ВРЕМЕНИ
-- =============================================================================

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_equipment_updated_at
    BEFORE UPDATE ON equipment
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_maintenance_events_updated_at
    BEFORE UPDATE ON maintenance_events
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at
    BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
