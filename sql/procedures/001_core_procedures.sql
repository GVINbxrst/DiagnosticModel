-- =============================================================================
-- DiagMod Database Procedures - Business Logic Functions
-- Хранимые процедуры для бизнес-логики и оптимизации операций
-- =============================================================================

SET search_path TO diagmod, public;

-- =============================================================================
-- ПРОЦЕДУРЫ ДЛЯ РАБОТЫ С СИГНАЛАМИ
-- =============================================================================

-- Функция для декодирования BYTEA в массив float32
CREATE OR REPLACE FUNCTION decode_float32_array(data BYTEA)
RETURNS REAL[] AS $$
DECLARE
    result REAL[];
    i INTEGER;
    byte_length INTEGER;
BEGIN
    IF data IS NULL THEN
        RETURN NULL;
    END IF;

    byte_length := length(data);

    -- Проверяем, что длина кратна 4 (размер float32)
    IF byte_length % 4 != 0 THEN
        RAISE EXCEPTION 'Invalid BYTEA length for float32 array: %', byte_length;
    END IF;

    -- Инициализируем массив
    result := ARRAY[]::REAL[];

    -- Декодируем по 4 байта в float32
    FOR i IN 0..(byte_length/4 - 1) LOOP
        result := array_append(result,
            (get_byte(data, i*4) |
             (get_byte(data, i*4+1) << 8) |
             (get_byte(data, i*4+2) << 16) |
             (get_byte(data, i*4+3) << 24))::INTEGER::REAL);
    END LOOP;

    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Функция для кодирования массива float32 в BYTEA
CREATE OR REPLACE FUNCTION encode_float32_array(data REAL[])
RETURNS BYTEA AS $$
DECLARE
    result BYTEA := '\x';
    val INTEGER;
    i INTEGER;
BEGIN
    IF data IS NULL OR array_length(data, 1) IS NULL THEN
        RETURN NULL;
    END IF;

    FOR i IN 1..array_length(data, 1) LOOP
        val := data[i]::INTEGER;
        result := result ||
                  chr(val & 255)::BYTEA ||
                  chr((val >> 8) & 255)::BYTEA ||
                  chr((val >> 16) & 255)::BYTEA ||
                  chr((val >> 24) & 255)::BYTEA;
    END LOOP;

    RETURN result;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Процедура для вставки нового сигнала с проверкой дубликатов
CREATE OR REPLACE FUNCTION insert_raw_signal(
    p_equipment_id UUID,
    p_recorded_at TIMESTAMPTZ,
    p_sample_rate_hz INTEGER,
    p_samples_count INTEGER,
    p_phase_a REAL[] DEFAULT NULL,
    p_phase_b REAL[] DEFAULT NULL,
    p_phase_c REAL[] DEFAULT NULL,
    p_meta JSONB DEFAULT '{}',
    p_file_name VARCHAR(500) DEFAULT NULL,
    p_file_hash VARCHAR(64) DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    signal_id UUID;
    existing_id UUID;
BEGIN
    -- Проверка на дубликат по хешу файла
    IF p_file_hash IS NOT NULL THEN
        SELECT id INTO existing_id
        FROM raw_signals
        WHERE file_hash = p_file_hash;

        IF existing_id IS NOT NULL THEN
            RAISE NOTICE 'Signal with hash % already exists: %', p_file_hash, existing_id;
            RETURN existing_id;
        END IF;
    END IF;

    -- Валидация данных
    IF p_phase_a IS NULL AND p_phase_b IS NULL AND p_phase_c IS NULL THEN
        RAISE EXCEPTION 'At least one phase must contain data';
    END IF;

    IF p_sample_rate_hz <= 0 THEN
        RAISE EXCEPTION 'Sample rate must be positive';
    END IF;

    IF p_samples_count <= 0 THEN
        RAISE EXCEPTION 'Samples count must be positive';
    END IF;

    -- Вставка записи
    INSERT INTO raw_signals (
        equipment_id,
        recorded_at,
        sample_rate_hz,
        samples_count,
        phase_a,
        phase_b,
        phase_c,
        meta,
        file_name,
        file_hash
    ) VALUES (
        p_equipment_id,
        p_recorded_at,
        p_sample_rate_hz,
        p_samples_count,
        encode_float32_array(p_phase_a),
        encode_float32_array(p_phase_b),
        encode_float32_array(p_phase_c),
        p_meta,
        p_file_name,
        p_file_hash
    ) RETURNING id INTO signal_id;

    RETURN signal_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ПРОЦЕДУРЫ ДЛЯ РАБОТЫ С ПРИЗНАКАМИ
-- =============================================================================

-- Процедура для вставки признаков
CREATE OR REPLACE FUNCTION insert_features(
    p_raw_id UUID,
    p_window_start TIMESTAMPTZ,
    p_window_end TIMESTAMPTZ,
    p_features JSONB
) RETURNS UUID AS $$
DECLARE
    feature_id UUID;
    existing_id UUID;
BEGIN
    -- Проверка на дубликат
    SELECT id INTO existing_id
    FROM features
    WHERE raw_id = p_raw_id
      AND window_start = p_window_start
      AND window_end = p_window_end;

    IF existing_id IS NOT NULL THEN
        RAISE NOTICE 'Features for this window already exist: %', existing_id;
        RETURN existing_id;
    END IF;

    -- Валидация временного окна
    IF p_window_end <= p_window_start THEN
        RAISE EXCEPTION 'Window end must be after window start';
    END IF;

    -- Вставка признаков
    INSERT INTO features (
        raw_id,
        window_start,
        window_end,
        rms_a, rms_b, rms_c,
        crest_a, crest_b, crest_c,
        kurt_a, kurt_b, kurt_c,
        skew_a, skew_b, skew_c,
        mean_a, mean_b, mean_c,
        std_a, std_b, std_c,
        min_a, min_b, min_c,
        max_a, max_b, max_c,
        fft_spectrum,
        extra
    ) VALUES (
        p_raw_id,
        p_window_start,
        p_window_end,
        (p_features->>'rms_a')::REAL,
        (p_features->>'rms_b')::REAL,
        (p_features->>'rms_c')::REAL,
        (p_features->>'crest_a')::REAL,
        (p_features->>'crest_b')::REAL,
        (p_features->>'crest_c')::REAL,
        (p_features->>'kurt_a')::REAL,
        (p_features->>'kurt_b')::REAL,
        (p_features->>'kurt_c')::REAL,
        (p_features->>'skew_a')::REAL,
        (p_features->>'skew_b')::REAL,
        (p_features->>'skew_c')::REAL,
        (p_features->>'mean_a')::REAL,
        (p_features->>'mean_b')::REAL,
        (p_features->>'mean_c')::REAL,
        (p_features->>'std_a')::REAL,
        (p_features->>'std_b')::REAL,
        (p_features->>'std_c')::REAL,
        (p_features->>'min_a')::REAL,
        (p_features->>'min_b')::REAL,
        (p_features->>'min_c')::REAL,
        (p_features->>'max_a')::REAL,
        (p_features->>'max_b')::REAL,
        (p_features->>'max_c')::REAL,
        p_features->'fft_spectrum',
        p_features->'extra'
    ) RETURNING id INTO feature_id;

    RETURN feature_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ПРОЦЕДУРЫ ДЛЯ АНАЛИЗА АНОМАЛИЙ
-- =============================================================================

-- Функция для расчета статистических аномалий по RMS
CREATE OR REPLACE FUNCTION detect_rms_anomalies(
    p_equipment_id UUID,
    p_lookback_days INTEGER DEFAULT 30,
    p_sigma_threshold REAL DEFAULT 3.0
) RETURNS TABLE (
    feature_id UUID,
    window_start TIMESTAMPTZ,
    phase VARCHAR(1),
    rms_value REAL,
    mean_value REAL,
    std_value REAL,
    z_score REAL,
    is_anomaly BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH rms_stats AS (
        -- Расчет статистики по RMS для каждой фазы
        SELECT
            'A' as phase,
            AVG(rms_a) as mean_rms,
            STDDEV(rms_a) as std_rms
        FROM features f
        JOIN raw_signals rs ON f.raw_id = rs.id
        WHERE rs.equipment_id = p_equipment_id
          AND f.window_start > CURRENT_TIMESTAMP - (p_lookback_days || ' days')::INTERVAL
          AND rms_a IS NOT NULL

        UNION ALL

        SELECT
            'B' as phase,
            AVG(rms_b) as mean_rms,
            STDDEV(rms_b) as std_rms
        FROM features f
        JOIN raw_signals rs ON f.raw_id = rs.id
        WHERE rs.equipment_id = p_equipment_id
          AND f.window_start > CURRENT_TIMESTAMP - (p_lookback_days || ' days')::INTERVAL
          AND rms_b IS NOT NULL

        UNION ALL

        SELECT
            'C' as phase,
            AVG(rms_c) as mean_rms,
            STDDEV(rms_c) as std_rms
        FROM features f
        JOIN raw_signals rs ON f.raw_id = rs.id
        WHERE rs.equipment_id = p_equipment_id
          AND f.window_start > CURRENT_TIMESTAMP - (p_lookback_days || ' days')::INTERVAL
          AND rms_c IS NOT NULL
    ),
    recent_features AS (
        -- Последние признаки для анализа
        SELECT
            f.id,
            f.window_start,
            f.rms_a,
            f.rms_b,
            f.rms_c
        FROM features f
        JOIN raw_signals rs ON f.raw_id = rs.id
        WHERE rs.equipment_id = p_equipment_id
          AND f.window_start > CURRENT_TIMESTAMP - INTERVAL '24 hours'
    )
    -- Анализ аномалий по фазам
    SELECT
        rf.id,
        rf.window_start,
        'A'::VARCHAR(1),
        rf.rms_a,
        stats.mean_rms,
        stats.std_rms,
        (rf.rms_a - stats.mean_rms) / NULLIF(stats.std_rms, 0) as z_score,
        ABS((rf.rms_a - stats.mean_rms) / NULLIF(stats.std_rms, 0)) > p_sigma_threshold
    FROM recent_features rf
    CROSS JOIN rms_stats stats
    WHERE stats.phase = 'A' AND rf.rms_a IS NOT NULL

    UNION ALL

    SELECT
        rf.id,
        rf.window_start,
        'B'::VARCHAR(1),
        rf.rms_b,
        stats.mean_rms,
        stats.std_rms,
        (rf.rms_b - stats.mean_rms) / NULLIF(stats.std_rms, 0) as z_score,
        ABS((rf.rms_b - stats.mean_rms) / NULLIF(stats.std_rms, 0)) > p_sigma_threshold
    FROM recent_features rf
    CROSS JOIN rms_stats stats
    WHERE stats.phase = 'B' AND rf.rms_b IS NOT NULL

    UNION ALL

    SELECT
        rf.id,
        rf.window_start,
        'C'::VARCHAR(1),
        rf.rms_c,
        stats.mean_rms,
        stats.std_rms,
        (rf.rms_c - stats.mean_rms) / NULLIF(stats.std_rms, 0) as z_score,
        ABS((rf.rms_c - stats.mean_rms) / NULLIF(stats.std_rms, 0)) > p_sigma_threshold
    FROM recent_features rf
    CROSS JOIN rms_stats stats
    WHERE stats.phase = 'C' AND rf.rms_c IS NOT NULL

    ORDER BY window_start DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ПРОЦЕДУРЫ ДЛЯ ОБСЛУЖИВАНИЯ ДАННЫХ
-- =============================================================================

-- Процедура для очистки старых данных
CREATE OR REPLACE FUNCTION cleanup_old_data(
    p_retention_days INTEGER DEFAULT 365
) RETURNS TABLE (
    table_name TEXT,
    deleted_count BIGINT
) AS $$
DECLARE
    cutoff_date TIMESTAMPTZ;
    deleted_signals BIGINT;
    deleted_logs BIGINT;
    deleted_sessions BIGINT;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - (p_retention_days || ' days')::INTERVAL;

    -- Удаление старых сигналов (каскадно удалит связанные features и predictions)
    DELETE FROM raw_signals WHERE recorded_at < cutoff_date;
    GET DIAGNOSTICS deleted_signals = ROW_COUNT;

    -- Удаление старых системных логов
    DELETE FROM system_logs WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_logs = ROW_COUNT;

    -- Удаление просроченных сессий
    DELETE FROM user_sessions WHERE expires_at < CURRENT_TIMESTAMP;
    GET DIAGNOSTICS deleted_sessions = ROW_COUNT;

    -- Обновление статистики
    ANALYZE raw_signals;
    ANALYZE features;
    ANALYZE predictions;
    ANALYZE system_logs;
    ANALYZE user_sessions;

    -- Возврат результатов
    RETURN QUERY VALUES
        ('raw_signals', deleted_signals),
        ('system_logs', deleted_logs),
        ('user_sessions', deleted_sessions);
END;
$$ LANGUAGE plpgsql;

-- Процедура для обновления материализованных представлений
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS VOID AS $$
BEGIN
    -- Обновление дашборда
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_summary;

    -- Логирование
    INSERT INTO system_logs (level, module, message, details)
    VALUES ('INFO', 'maintenance', 'Materialized views refreshed',
            jsonb_build_object('timestamp', CURRENT_TIMESTAMP));
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- ПРОЦЕДУРЫ ДЛЯ СТАТИСТИКИ И ОТЧЕТНОСТИ
-- =============================================================================

-- Функция для получения сводки по оборудованию
CREATE OR REPLACE FUNCTION get_equipment_health_summary(
    p_equipment_id UUID DEFAULT NULL
) RETURNS TABLE (
    equipment_id_code VARCHAR(50),
    equipment_name VARCHAR(255),
    status equipment_status,
    last_signal_at TIMESTAMPTZ,
    total_signals BIGINT,
    avg_rms_24h REAL,
    max_risk_24h REAL,
    critical_alerts_24h BIGINT,
    next_maintenance_at TIMESTAMPTZ,
    health_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.equipment_id,
        e.name,
        e.status,

        -- Последний сигнал
        (SELECT MAX(rs.recorded_at)
         FROM raw_signals rs
         WHERE rs.equipment_id = e.id),

        -- Общее количество сигналов
        (SELECT COUNT(*)
         FROM raw_signals rs
         WHERE rs.equipment_id = e.id),

        -- Средний RMS за 24 часа
        (SELECT AVG(COALESCE(f.rms_a, 0) + COALESCE(f.rms_b, 0) + COALESCE(f.rms_c, 0)) /
                NULLIF((CASE WHEN f.rms_a IS NOT NULL THEN 1 ELSE 0 END +
                        CASE WHEN f.rms_b IS NOT NULL THEN 1 ELSE 0 END +
                        CASE WHEN f.rms_c IS NOT NULL THEN 1 ELSE 0 END), 0)
         FROM features f
         JOIN raw_signals rs ON f.raw_id = rs.id
         WHERE rs.equipment_id = e.id
           AND f.window_start > CURRENT_TIMESTAMP - INTERVAL '24 hours'),

        -- Максимальный риск за 24 часа
        (SELECT MAX(p.probability)
         FROM predictions p
         JOIN features f ON p.feature_id = f.id
         JOIN raw_signals rs ON f.raw_id = rs.id
         WHERE rs.equipment_id = e.id
           AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'),

        -- Критические алерты за 24 часа
        (SELECT COUNT(*)
         FROM predictions p
         JOIN features f ON p.feature_id = f.id
         JOIN raw_signals rs ON f.raw_id = rs.id
         WHERE rs.equipment_id = e.id
           AND p.predicted_severity = 'critical'
           AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'),

        -- Следующее обслуживание
        (SELECT MIN(me.scheduled_date)
         FROM maintenance_events me
         WHERE me.equipment_id = e.id
           AND me.status = 'scheduled'
           AND me.scheduled_date > CURRENT_TIMESTAMP),

        -- Health Score (0-100, где 100 = отлично)
        GREATEST(0, LEAST(100,
            100 -
            COALESCE((SELECT MAX(p.probability) * 100
                     FROM predictions p
                     JOIN features f ON p.feature_id = f.id
                     JOIN raw_signals rs ON f.raw_id = rs.id
                     WHERE rs.equipment_id = e.id
                       AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'), 0) -
            COALESCE((SELECT COUNT(*) * 10
                     FROM predictions p
                     JOIN features f ON p.feature_id = f.id
                     JOIN raw_signals rs ON f.raw_id = rs.id
                     WHERE rs.equipment_id = e.id
                       AND p.predicted_severity = 'critical'
                       AND p.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'), 0)
        ))::REAL
    FROM equipment e
    WHERE (p_equipment_id IS NULL OR e.id = p_equipment_id)
      AND e.status != 'inactive'
    ORDER BY health_score ASC, e.name;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- КОММЕНТАРИИ К ПРОЦЕДУРАМ
-- =============================================================================

COMMENT ON FUNCTION decode_float32_array(BYTEA) IS 'Декодирование BYTEA в массив float32';
COMMENT ON FUNCTION encode_float32_array(REAL[]) IS 'Кодирование массива float32 в BYTEA';
COMMENT ON FUNCTION insert_raw_signal(UUID, TIMESTAMPTZ, INTEGER, INTEGER, REAL[], REAL[], REAL[], JSONB, VARCHAR, VARCHAR) IS 'Вставка нового сигнала с проверкой дубликатов';
COMMENT ON FUNCTION insert_features(UUID, TIMESTAMPTZ, TIMESTAMPTZ, JSONB) IS 'Вставка извлеченных признаков';
COMMENT ON FUNCTION detect_rms_anomalies(UUID, INTEGER, REAL) IS 'Обнаружение аномалий по RMS с использованием статистических методов';
COMMENT ON FUNCTION cleanup_old_data(INTEGER) IS 'Очистка старых данных с заданным периодом хранения';
COMMENT ON FUNCTION refresh_materialized_views() IS 'Обновление всех материализованных представлений';
COMMENT ON FUNCTION get_equipment_health_summary(UUID) IS 'Получение сводки состояния оборудования';
