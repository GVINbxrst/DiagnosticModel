-- =============================================================================
-- DiagMod Database Views - Data Access Layer
-- Представления для удобного доступа к данным
-- =============================================================================

SET search_path TO diagmod, public;

-- =============================================================================
-- ПРЕДСТАВЛЕНИЯ ДЛЯ АНАЛИЗА ДАННЫХ
-- =============================================================================

-- Представление для получения сигналов с информацией об оборудовании
CREATE OR REPLACE VIEW v_signals_with_equipment AS
SELECT
    rs.id,
    rs.equipment_id,
    e.equipment_id as equipment_code,
    e.name as equipment_name,
    e.type as equipment_type,
    e.status as equipment_status,
    e.location,
    rs.recorded_at,
    rs.sample_rate_hz,
    rs.samples_count,
    rs.duration_ms,
    rs.phase_a IS NOT NULL as has_phase_a,
    rs.phase_b IS NOT NULL as has_phase_b,
    rs.phase_c IS NOT NULL as has_phase_c,
    rs.meta,
    rs.file_name,
    rs.processed,
    rs.created_at
FROM raw_signals rs
JOIN equipment e ON rs.equipment_id = e.id;

-- Представление для анализа признаков с контекстом
CREATE OR REPLACE VIEW v_features_analysis AS
SELECT
    f.id,
    f.raw_id,
    rs.equipment_id,
    e.equipment_id as equipment_code,
    e.name as equipment_name,
    f.window_start,
    f.window_end,
    f.window_size_ms,

    -- RMS анализ
    f.rms_a,
    f.rms_b,
    f.rms_c,
    COALESCE(f.rms_a, 0) + COALESCE(f.rms_b, 0) + COALESCE(f.rms_c, 0) as rms_total,
    (COALESCE(f.rms_a, 0) + COALESCE(f.rms_b, 0) + COALESCE(f.rms_c, 0)) /
    NULLIF((CASE WHEN f.rms_a IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN f.rms_b IS NOT NULL THEN 1 ELSE 0 END +
            CASE WHEN f.rms_c IS NOT NULL THEN 1 ELSE 0 END), 0) as rms_avg,

    -- Crest Factor анализ
    f.crest_a,
    f.crest_b,
    f.crest_c,
    GREATEST(COALESCE(f.crest_a, 0), COALESCE(f.crest_b, 0), COALESCE(f.crest_c, 0)) as crest_max,

    -- Статистические моменты
    f.kurt_a,
    f.kurt_b,
    f.kurt_c,
    f.skew_a,
    f.skew_b,
    f.skew_c,

    -- Базовая статистика
    f.mean_a,
    f.mean_b,
    f.mean_c,
    f.std_a,
    f.std_b,
    f.std_c,

    -- Диапазоны
    f.max_a - f.min_a as range_a,
    f.max_b - f.min_b as range_b,
    f.max_c - f.min_c as range_c,

    f.fft_spectrum,
    f.extra,
    f.created_at
FROM features f
JOIN raw_signals rs ON f.raw_id = rs.id
JOIN equipment e ON rs.equipment_id = e.id;

-- Представление для анализа предсказаний
CREATE OR REPLACE VIEW v_predictions_detailed AS
SELECT
    p.id,
    p.feature_id,
    f.raw_id,
    rs.equipment_id,
    e.equipment_id as equipment_code,
    e.name as equipment_name,
    f.window_start,
    f.window_end,

    -- Информация о дефекте
    p.defect_type_id,
    dt.code as defect_code,
    dt.name as defect_name,
    dt.category as defect_category,

    -- Результаты предсказания
    p.probability,
    p.predicted_severity,
    p.confidence_score,

    -- Информация о модели
    p.model_name,
    p.model_version,
    p.model_type,

    p.prediction_details,
    p.created_at
FROM predictions p
JOIN features f ON p.feature_id = f.id
JOIN raw_signals rs ON f.raw_id = rs.id
JOIN equipment e ON rs.equipment_id = e.id
LEFT JOIN defect_types dt ON p.defect_type_id = dt.id;

-- =============================================================================
-- АНАЛИТИЧЕСКИЕ ПРЕДСТАВЛЕНИЯ
-- =============================================================================

-- Сводка по оборудованию с последними данными
CREATE OR REPLACE VIEW v_equipment_summary AS
SELECT
    e.id,
    e.equipment_id,
    e.name,
    e.type,
    e.status,
    e.location,

    -- Статистика сигналов
    COUNT(rs.id) as total_signals,
    MIN(rs.recorded_at) as first_signal_at,
    MAX(rs.recorded_at) as last_signal_at,
    COUNT(CASE WHEN rs.processed = false THEN 1 END) as unprocessed_signals,

    -- Статистика признаков
    COUNT(f.id) as total_features,

    -- Статистика предсказаний
    COUNT(p.id) as total_predictions,
    COUNT(CASE WHEN p.probability > 0.7 THEN 1 END) as high_risk_predictions,
    COUNT(CASE WHEN p.predicted_severity = 'critical' THEN 1 END) as critical_predictions,

    -- Последнее обслуживание
    (SELECT MAX(completed_at)
     FROM maintenance_events me
     WHERE me.equipment_id = e.id AND me.status = 'completed') as last_maintenance_at,

    e.created_at,
    e.updated_at
FROM equipment e
LEFT JOIN raw_signals rs ON e.id = rs.equipment_id
LEFT JOIN features f ON rs.id = f.raw_id
LEFT JOIN predictions p ON f.id = p.feature_id
GROUP BY e.id, e.equipment_id, e.name, e.type, e.status, e.location, e.created_at, e.updated_at;

-- Тренды RMS по оборудованию (последние 30 дней)
CREATE OR REPLACE VIEW v_rms_trends AS
SELECT
    e.equipment_id,
    e.name as equipment_name,
    DATE_TRUNC('day', f.window_start) as trend_date,

    -- Агрегированные RMS значения за день
    AVG(f.rms_a) as avg_rms_a,
    AVG(f.rms_b) as avg_rms_b,
    AVG(f.rms_c) as avg_rms_c,

    MIN(f.rms_a) as min_rms_a,
    MIN(f.rms_b) as min_rms_b,
    MIN(f.rms_c) as min_rms_c,

    MAX(f.rms_a) as max_rms_a,
    MAX(f.rms_b) as max_rms_b,
    MAX(f.rms_c) as max_rms_c,

    STDDEV(f.rms_a) as std_rms_a,
    STDDEV(f.rms_b) as std_rms_b,
    STDDEV(f.rms_c) as std_rms_c,

    COUNT(*) as measurements_count
FROM features f
JOIN raw_signals rs ON f.raw_id = rs.id
JOIN equipment e ON rs.equipment_id = e.id
WHERE f.window_start > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY e.equipment_id, e.name, DATE_TRUNC('day', f.window_start)
ORDER BY e.equipment_id, trend_date DESC;

-- Анализ аномалий по критичности
CREATE OR REPLACE VIEW v_anomaly_analysis AS
SELECT
    e.equipment_id,
    e.name as equipment_name,
    dt.category as defect_category,
    p.predicted_severity,

    -- Счетчики за разные периоды
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END) as last_24h,
    COUNT(CASE WHEN p.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as last_7d,
    COUNT(CASE WHEN p.created_at > CURRENT_TIMESTAMP - INTERVAL '30 days' THEN 1 END) as last_30d,

    -- Статистика вероятностей
    AVG(p.probability) as avg_probability,
    MAX(p.probability) as max_probability,
    MIN(p.probability) as min_probability,

    -- Последнее предсказание
    MAX(p.created_at) as last_prediction_at
FROM predictions p
JOIN features f ON p.feature_id = f.id
JOIN raw_signals rs ON f.raw_id = rs.id
JOIN equipment e ON rs.equipment_id = e.id
LEFT JOIN defect_types dt ON p.defect_type_id = dt.id
WHERE p.probability > 0.5  -- Только значимые предсказания
GROUP BY e.equipment_id, e.name, dt.category, p.predicted_severity
ORDER BY max_probability DESC, last_prediction_at DESC;

-- =============================================================================
-- ПРЕДСТАВЛЕНИЯ ДЛЯ МОНИТОРИНГА
-- =============================================================================

-- Общая статистика системы
CREATE OR REPLACE VIEW v_system_health AS
SELECT
    -- Статистика данных
    (SELECT COUNT(*) FROM raw_signals) as total_signals,
    (SELECT COUNT(*) FROM raw_signals WHERE processed = false) as unprocessed_signals,
    (SELECT COUNT(*) FROM features) as total_features,
    (SELECT COUNT(*) FROM predictions) as total_predictions,

    -- Статистика оборудования
    (SELECT COUNT(*) FROM equipment) as total_equipment,
    (SELECT COUNT(*) FROM equipment WHERE status = 'active') as active_equipment,
    (SELECT COUNT(*) FROM equipment WHERE status = 'fault') as faulty_equipment,

    -- Последние данные
    (SELECT MAX(recorded_at) FROM raw_signals) as last_signal_at,
    (SELECT MAX(created_at) FROM features) as last_feature_at,
    (SELECT MAX(created_at) FROM predictions) as last_prediction_at,

    -- Критические предсказания (последние 24 часа)
    (SELECT COUNT(*)
     FROM predictions
     WHERE predicted_severity = 'critical'
       AND created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as critical_alerts_24h,

    -- Активные пользователи
    (SELECT COUNT(*) FROM users WHERE is_active = true) as active_users,

    -- Запланированное обслуживание
    (SELECT COUNT(*)
     FROM maintenance_events
     WHERE status = 'scheduled'
       AND scheduled_date BETWEEN CURRENT_TIMESTAMP AND CURRENT_TIMESTAMP + INTERVAL '7 days') as scheduled_maintenance_7d;

-- Производительность системы
CREATE OR REPLACE VIEW v_performance_metrics AS
SELECT
    -- Объем данных по дням (последние 30 дней)
    DATE_TRUNC('day', created_at) as metric_date,
    'signals' as metric_type,
    COUNT(*) as count,
    AVG(samples_count) as avg_samples,
    SUM(CASE WHEN phase_a IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN phase_b IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN phase_c IS NOT NULL THEN 1 ELSE 0 END) as total_phases
FROM raw_signals
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)

UNION ALL

SELECT
    DATE_TRUNC('day', created_at) as metric_date,
    'features' as metric_type,
    COUNT(*) as count,
    AVG(window_size_ms) as avg_samples,
    NULL as total_phases
FROM features
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)

UNION ALL

SELECT
    DATE_TRUNC('day', created_at) as metric_date,
    'predictions' as metric_type,
    COUNT(*) as count,
    AVG(probability) as avg_samples,
    NULL as total_phases
FROM predictions
WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)

ORDER BY metric_date DESC, metric_type;

-- =============================================================================
-- МАТЕРИАЛИЗОВАННЫЕ ПРЕДСТАВЛЕНИЯ ДЛЯ БЫСТРОГО ДОСТУПА
-- =============================================================================

-- Материализованное представление для дашборда (обновляется каждый час)
CREATE MATERIALIZED VIEW mv_dashboard_summary AS
SELECT
    e.equipment_id,
    e.name as equipment_name,
    e.type as equipment_type,
    e.status as equipment_status,
    e.location,

    -- Последние данные
    (SELECT rs.recorded_at
     FROM raw_signals rs
     WHERE rs.equipment_id = e.id
     ORDER BY rs.recorded_at DESC
     LIMIT 1) as last_signal_at,

    -- Последние признаки (средние RMS за последние 24 часа)
    (SELECT AVG(f.rms_a)
     FROM features f
     JOIN raw_signals rs ON f.raw_id = rs.id
     WHERE rs.equipment_id = e.id
       AND f.window_start > CURRENT_TIMESTAMP - INTERVAL '24 hours') as avg_rms_a_24h,

    (SELECT AVG(f.rms_b)
     FROM features f
     JOIN raw_signals rs ON f.raw_id = rs.id
     WHERE rs.equipment_id = e.id
       AND f.window_start > CURRENT_TIMESTAMP - INTERVAL '24 hours') as avg_rms_b_24h,

    (SELECT AVG(f.rms_c)
     FROM features f
     JOIN raw_signals rs ON f.raw_id = rs.id
     WHERE rs.equipment_id = e.id
       AND f.window_start > CURRENT_TIMESTAMP - INTERVAL '24 hours') as avg_rms_c_24h,

    -- Последние предсказания
    (SELECT MAX(p.probability)
     FROM predictions p
     JOIN features f ON p.feature_id = f.id
     JOIN raw_signals rs ON f.raw_id = rs.id
     WHERE rs.equipment_id = e.id
       AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as max_risk_24h,

    (SELECT COUNT(*)
     FROM predictions p
     JOIN features f ON p.feature_id = f.id
     JOIN raw_signals rs ON f.raw_id = rs.id
     WHERE rs.equipment_id = e.id
       AND p.predicted_severity = 'critical'
       AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as critical_alerts_24h,

    -- Следующее обслуживание
    (SELECT MIN(me.scheduled_date)
     FROM maintenance_events me
     WHERE me.equipment_id = e.id
       AND me.status = 'scheduled'
       AND me.scheduled_date > CURRENT_TIMESTAMP) as next_maintenance_at,

    CURRENT_TIMESTAMP as last_updated
FROM equipment e
WHERE e.status != 'inactive';

-- Создание индекса для материализованного представления
CREATE INDEX idx_mv_dashboard_summary_equipment_id
    ON mv_dashboard_summary (equipment_id);

-- =============================================================================
-- КОММЕНТАРИИ К ПРЕДСТАВЛЕНИЯМ
-- =============================================================================

COMMENT ON VIEW v_signals_with_equipment IS 'Сигналы с информацией об оборудовании';
COMMENT ON VIEW v_features_analysis IS 'Подробный анализ извлеченных признаков';
COMMENT ON VIEW v_predictions_detailed IS 'Детализированные предсказания с контекстом';
COMMENT ON VIEW v_equipment_summary IS 'Сводная информация по оборудованию';
COMMENT ON VIEW v_rms_trends IS 'Тренды RMS значений по дням';
COMMENT ON VIEW v_anomaly_analysis IS 'Анализ аномалий по критичности';
COMMENT ON VIEW v_system_health IS 'Общее состояние системы';
COMMENT ON VIEW v_performance_metrics IS 'Метрики производительности системы';
COMMENT ON MATERIALIZED VIEW mv_dashboard_summary IS 'Материализованная сводка для дашборда (обновляется каждый час)';
