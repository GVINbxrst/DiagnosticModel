-- =============================================================================
-- DiagMod Database Indexes - Performance Optimization
-- Создание индексов для оптимизации запросов
-- =============================================================================

SET search_path TO diagmod, public;

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ raw_signals
-- =============================================================================

-- Основной временной индекс (критичен для временных запросов)
CREATE INDEX CONCURRENTLY idx_raw_signals_recorded_at
    ON raw_signals USING BTREE (recorded_at DESC);

-- Индекс по оборудованию и времени (для запросов по конкретному оборудованию)
CREATE INDEX CONCURRENTLY idx_raw_signals_equipment_time
    ON raw_signals USING BTREE (equipment_id, recorded_at DESC);

-- Индекс для неprocessed записей (для фоновой обработки)
CREATE INDEX CONCURRENTLY idx_raw_signals_unprocessed
    ON raw_signals USING BTREE (processed, created_at)
    WHERE processed = false;

-- Индекс по статусу обработки pipeline
CREATE INDEX CONCURRENTLY idx_raw_signals_processing_status
    ON raw_signals USING BTREE (processing_status, created_at DESC);

-- Индекс по хешу файла (для дедупликации)
CREATE INDEX CONCURRENTLY idx_raw_signals_file_hash
    ON raw_signals USING BTREE (file_hash)
    WHERE file_hash IS NOT NULL;

-- GIN индекс для поиска по метаданным
CREATE INDEX CONCURRENTLY idx_raw_signals_meta
    ON raw_signals USING GIN (meta);

-- Партиционированный индекс по месяцам (для архивирования)
CREATE INDEX CONCURRENTLY idx_raw_signals_monthly
    ON raw_signals USING BTREE (DATE_TRUNC('month', recorded_at), equipment_id);

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ features
-- =============================================================================

-- Связь с raw_signals
CREATE INDEX CONCURRENTLY idx_features_raw_id
    ON features USING BTREE (raw_id);

-- Временные индексы для анализа трендов
CREATE INDEX CONCURRENTLY idx_features_window_start
    ON features USING BTREE (window_start DESC);

CREATE INDEX CONCURRENTLY idx_features_window_range
    ON features USING BTREE (window_start, window_end);

-- Комбинированный индекс для поиска признаков по оборудованию
CREATE INDEX CONCURRENTLY idx_features_equipment_time
    ON features USING BTREE (
        (SELECT equipment_id FROM raw_signals WHERE id = features.raw_id),
        window_start DESC
    );

-- Индексы для статистических признаков (для быстрого поиска аномалий)
CREATE INDEX CONCURRENTLY idx_features_rms_a
    ON features USING BTREE (rms_a)
    WHERE rms_a IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_features_rms_b
    ON features USING BTREE (rms_b)
    WHERE rms_b IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_features_rms_c
    ON features USING BTREE (rms_c)
    WHERE rms_c IS NOT NULL;

-- Индексы для поиска выбросов
CREATE INDEX CONCURRENTLY idx_features_crest_factors
    ON features USING BTREE (crest_a, crest_b, crest_c);

CREATE INDEX CONCURRENTLY idx_features_kurtosis
    ON features USING BTREE (kurt_a, kurt_b, kurt_c);

-- GIN индекс для FFT спектра
CREATE INDEX CONCURRENTLY idx_features_fft_spectrum
    ON features USING GIN (fft_spectrum);

-- GIN индекс для дополнительных признаков
CREATE INDEX CONCURRENTLY idx_features_extra
    ON features USING GIN (extra);

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ predictions
-- =============================================================================

-- Связь с features
CREATE INDEX CONCURRENTLY idx_predictions_feature_id
    ON predictions USING BTREE (feature_id);

-- Индекс по типу дефекта
CREATE INDEX CONCURRENTLY idx_predictions_defect_type
    ON predictions USING BTREE (defect_type_id);

-- Индекс по вероятности (для поиска высоковероятных аномалий)
CREATE INDEX CONCURRENTLY idx_predictions_high_probability
    ON predictions USING BTREE (probability DESC)
    WHERE probability > 0.7;

-- Индекс по оборудованию (частые фильтры)
CREATE INDEX CONCURRENTLY idx_predictions_equipment
    ON predictions USING BTREE (equipment_id, created_at DESC);

-- Индекс по флагу аномалии
CREATE INDEX CONCURRENTLY idx_predictions_anomaly_detected
    ON predictions USING BTREE (anomaly_detected, confidence DESC, created_at DESC)
    WHERE anomaly_detected = true;

-- Индекс по confidence (общие фильтры)
CREATE INDEX CONCURRENTLY idx_predictions_confidence
    ON predictions USING BTREE (confidence DESC)
    WHERE confidence > 0.0;

-- Индекс по критичности
CREATE INDEX CONCURRENTLY idx_predictions_severity
    ON predictions USING BTREE (predicted_severity);

-- Комбинированный индекс для анализа моделей
CREATE INDEX CONCURRENTLY idx_predictions_model_analysis
    ON predictions USING BTREE (model_name, model_version, created_at DESC);

-- Временной индекс для последних предсказаний
CREATE INDEX CONCURRENTLY idx_predictions_created_at
    ON predictions USING BTREE (created_at DESC);

-- GIN индекс для деталей предсказания
CREATE INDEX CONCURRENTLY idx_predictions_details
    ON predictions USING GIN (prediction_details);

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ equipment
-- =============================================================================

-- Уникальный индекс по equipment_id
CREATE UNIQUE INDEX CONCURRENTLY idx_equipment_equipment_id
    ON equipment USING BTREE (equipment_id);

-- Индекс по статусу (для быстрого поиска активного оборудования)
CREATE INDEX CONCURRENTLY idx_equipment_status
    ON equipment USING BTREE (status);

-- Индекс по типу оборудования
CREATE INDEX CONCURRENTLY idx_equipment_type
    ON equipment USING BTREE (type);

-- Текстовый поиск по названию и местоположению
CREATE INDEX CONCURRENTLY idx_equipment_name_search
    ON equipment USING GIN (to_tsvector('russian', name || ' ' || COALESCE(location, '')));

-- GIN индекс для спецификаций
CREATE INDEX CONCURRENTLY idx_equipment_specifications
    ON equipment USING GIN (specifications);

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ maintenance_events
-- =============================================================================

-- Связь с оборудованием
CREATE INDEX CONCURRENTLY idx_maintenance_equipment_id
    ON maintenance_events USING BTREE (equipment_id);

-- Временные индексы
CREATE INDEX CONCURRENTLY idx_maintenance_scheduled_date
    ON maintenance_events USING BTREE (scheduled_date)
    WHERE scheduled_date IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_maintenance_completed_at
    ON maintenance_events USING BTREE (completed_at DESC)
    WHERE completed_at IS NOT NULL;

-- Индекс по статусу
CREATE INDEX CONCURRENTLY idx_maintenance_status
    ON maintenance_events USING BTREE (status);

-- Комбинированный индекс для планирования
CREATE INDEX CONCURRENTLY idx_maintenance_planning
    ON maintenance_events USING BTREE (equipment_id, status, scheduled_date);

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ users
-- =============================================================================

-- Уникальные индексы (автоматически созданы, но явно указаны для ясности)
CREATE UNIQUE INDEX CONCURRENTLY idx_users_username
    ON users USING BTREE (username);

CREATE UNIQUE INDEX CONCURRENTLY idx_users_email
    ON users USING BTREE (email);

-- Индекс по роли
CREATE INDEX CONCURRENTLY idx_users_role
    ON users USING BTREE (role);

-- Индекс для активных пользователей
CREATE INDEX CONCURRENTLY idx_users_active
    ON users USING BTREE (is_active, last_login DESC)
    WHERE is_active = true;

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ ТАБЛИЦЫ defect_types
-- =============================================================================

-- Уникальный индекс по коду
CREATE UNIQUE INDEX CONCURRENTLY idx_defect_types_code
    ON defect_types USING BTREE (code);

-- Индекс по категории
CREATE INDEX CONCURRENTLY idx_defect_types_category
    ON defect_types USING BTREE (category);

-- Индекс для активных типов дефектов
CREATE INDEX CONCURRENTLY idx_defect_types_active
    ON defect_types USING BTREE (is_active, category)
    WHERE is_active = true;

-- =============================================================================
-- ИНДЕКСЫ ДЛЯ СЛУЖЕБНЫХ ТАБЛИЦ
-- =============================================================================

-- user_sessions
CREATE INDEX CONCURRENTLY idx_user_sessions_user_id
    ON user_sessions USING BTREE (user_id);

CREATE INDEX CONCURRENTLY idx_user_sessions_expires_at
    ON user_sessions USING BTREE (expires_at);

CREATE INDEX CONCURRENTLY idx_user_sessions_token_hash
    ON user_sessions USING BTREE (token_hash);

-- system_logs
CREATE INDEX CONCURRENTLY idx_system_logs_created_at
    ON system_logs USING BTREE (created_at DESC);

CREATE INDEX CONCURRENTLY idx_system_logs_level
    ON system_logs USING BTREE (level, created_at DESC);

CREATE INDEX CONCURRENTLY idx_system_logs_module
    ON system_logs USING BTREE (module, created_at DESC);

CREATE INDEX CONCURRENTLY idx_system_logs_user_id
    ON system_logs USING BTREE (user_id, created_at DESC)
    WHERE user_id IS NOT NULL;

-- GIN индекс для поиска по деталям логов
CREATE INDEX CONCURRENTLY idx_system_logs_details
    ON system_logs USING GIN (details);

-- system_config
CREATE UNIQUE INDEX CONCURRENTLY idx_system_config_key
    ON system_config USING BTREE (key);

-- =============================================================================
-- СПЕЦИАЛЬНЫЕ ИНДЕКСЫ ДЛЯ АНАЛИТИКИ
-- =============================================================================

-- Материализованное представление для быстрой аналитики
-- (будет создано в отдельном скрипте представлений)

-- Частичные индексы для критических аномалий
CREATE INDEX CONCURRENTLY idx_critical_predictions
    ON predictions USING BTREE (created_at DESC, probability DESC)
    WHERE predicted_severity = 'critical' AND probability > 0.8;

-- Индекс для недавних данных (последние 30 дней)
CREATE INDEX CONCURRENTLY idx_recent_raw_signals
    ON raw_signals USING BTREE (equipment_id, recorded_at DESC)
    WHERE recorded_at > CURRENT_TIMESTAMP - INTERVAL '30 days';

-- Индекс для анализа трендов по RMS
CREATE INDEX CONCURRENTLY idx_features_rms_trend
    ON features USING BTREE (
        (SELECT equipment_id FROM raw_signals WHERE id = features.raw_id),
        window_start,
        (COALESCE(rms_a, 0) + COALESCE(rms_b, 0) + COALESCE(rms_c, 0)) /
        NULLIF((CASE WHEN rms_a IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN rms_b IS NOT NULL THEN 1 ELSE 0 END +
                CASE WHEN rms_c IS NOT NULL THEN 1 ELSE 0 END), 0)
    );

-- =============================================================================
-- СТАТИСТИКА ДЛЯ ОПТИМИЗАТОРА
-- =============================================================================

-- Обновление статистики для лучшего планирования запросов
ANALYZE raw_signals;
ANALYZE features;
ANALYZE predictions;
ANALYZE equipment;
ANALYZE maintenance_events;
ANALYZE users;
ANALYZE defect_types;
