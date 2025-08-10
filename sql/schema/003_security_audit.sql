-- Создание неизменяемой таблицы для audit-логов
-- Эта таблица предназначена для безопасного хранения всех действий пользователей
-- с защитой от изменения и удаления данных

-- Создаем схему для безопасности если не существует
CREATE SCHEMA IF NOT EXISTS security;

-- Enum для типов действий аудита
CREATE TYPE security.audit_action_type AS ENUM (
    'login',
    'logout',
    'token_refresh',
    'file_upload',
    'signal_view',
    'anomaly_request',
    'forecast_request',
    'equipment_access',
    'admin_action',
    'data_export',
    'model_training',
    'system_config_change',
    'user_management',
    'failed_login',
    'permission_denied'
);

-- Enum для результатов действий
CREATE TYPE security.audit_result AS ENUM (
    'success',
    'failure',
    'denied',
    'error'
);

-- Основная таблица audit-логов
CREATE TABLE security.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Информация о пользователе
    user_id UUID NOT NULL,
    username VARCHAR(255) NOT NULL,
    user_role VARCHAR(50) NOT NULL,

    -- Информация о действии
    action_type security.audit_action_type NOT NULL,
    action_description TEXT NOT NULL,
    result security.audit_result NOT NULL,

    -- Детали запроса
    endpoint VARCHAR(500),
    http_method VARCHAR(10),
    request_ip INET NOT NULL,
    user_agent TEXT,

    -- Данные запроса и ответа (без чувствительной информации)
    request_data JSONB,
    response_status INTEGER,

    -- Информация о ресурсах
    resource_type VARCHAR(100), -- equipment, signal, model, etc.
    resource_id UUID,
    resource_name VARCHAR(255),

    -- Временные метки
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id VARCHAR(255),

    -- Метаданные
    additional_data JSONB,

    -- Хеш для проверки целостности записи
    integrity_hash VARCHAR(64) NOT NULL
);

-- Создаем индексы для производительности
CREATE INDEX idx_audit_logs_user_id ON security.audit_logs(user_id);
CREATE INDEX idx_audit_logs_timestamp ON security.audit_logs(timestamp);
CREATE INDEX idx_audit_logs_action_type ON security.audit_logs(action_type);
CREATE INDEX idx_audit_logs_result ON security.audit_logs(result);
CREATE INDEX idx_audit_logs_endpoint ON security.audit_logs(endpoint);
CREATE INDEX idx_audit_logs_ip ON security.audit_logs(request_ip);
CREATE INDEX idx_audit_logs_resource ON security.audit_logs(resource_type, resource_id);

-- Составной индекс для часто используемых запросов
CREATE INDEX idx_audit_logs_user_action_time ON security.audit_logs(user_id, action_type, timestamp DESC);

-- Функция для генерации хеша целостности
CREATE OR REPLACE FUNCTION security.generate_audit_hash(
    p_user_id UUID,
    p_action_type security.audit_action_type,
    p_timestamp TIMESTAMPTZ,
    p_request_ip INET,
    p_endpoint VARCHAR
) RETURNS VARCHAR(64) AS $$
BEGIN
    -- Создаем хеш SHA256 из ключевых полей записи
    RETURN encode(
        digest(
            CONCAT(
                p_user_id::text,
                p_action_type::text,
                p_timestamp::text,
                p_request_ip::text,
                COALESCE(p_endpoint, '')
            ),
            'sha256'
        ),
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Триггер для автоматического заполнения хеша целостности
CREATE OR REPLACE FUNCTION security.set_audit_integrity_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.integrity_hash = security.generate_audit_hash(
        NEW.user_id,
        NEW.action_type,
        NEW.timestamp,
        NEW.request_ip,
        NEW.endpoint
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Применяем триггер к таблице
CREATE TRIGGER trigger_audit_integrity_hash
    BEFORE INSERT ON security.audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION security.set_audit_integrity_hash();

-- Триггер для предотвращения изменения записей (делает таблицу неизменяемой)
CREATE OR REPLACE FUNCTION security.prevent_audit_modifications()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit logs are immutable. Operation % is not allowed.', TG_OP;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Запрещаем UPDATE и DELETE
CREATE TRIGGER trigger_prevent_audit_update
    BEFORE UPDATE ON security.audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION security.prevent_audit_modifications();

CREATE TRIGGER trigger_prevent_audit_delete
    BEFORE DELETE ON security.audit_logs
    FOR EACH ROW
    EXECUTE FUNCTION security.prevent_audit_modifications();

-- Таблица для хранения активных сессий
CREATE TABLE security.user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) NOT NULL UNIQUE,
    refresh_token_hash VARCHAR(255) NOT NULL,

    -- Информация о сессии
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,

    -- Информация о клиенте
    ip_address INET NOT NULL,
    user_agent TEXT,

    -- Флаги состояния
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    revoked_at TIMESTAMPTZ,
    revoked_reason VARCHAR(255)
);

-- Индексы для таблицы сессий
CREATE INDEX idx_user_sessions_user_id ON security.user_sessions(user_id);
CREATE INDEX idx_user_sessions_token ON security.user_sessions(session_token);
CREATE INDEX idx_user_sessions_active ON security.user_sessions(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_user_sessions_expires ON security.user_sessions(expires_at);

-- Представление для анализа audit-логов
CREATE VIEW security.v_audit_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    action_type,
    result,
    user_role,
    COUNT(*) as action_count,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT request_ip) as unique_ips
FROM security.audit_logs
GROUP BY DATE_TRUNC('hour', timestamp), action_type, result, user_role
ORDER BY hour DESC;

-- Представление для мониторинга подозрительной активности
CREATE VIEW security.v_suspicious_activity AS
SELECT
    user_id,
    username,
    request_ip,
    COUNT(*) as failed_attempts,
    MAX(timestamp) as last_attempt,
    MIN(timestamp) as first_attempt
FROM security.audit_logs
WHERE action_type IN ('failed_login', 'permission_denied')
    AND timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY user_id, username, request_ip
HAVING COUNT(*) >= 5  -- 5 или более неудачных попыток за час
ORDER BY failed_attempts DESC;

-- Функция для очистки старых сессий
CREATE OR REPLACE FUNCTION security.cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    cleaned_count INTEGER;
BEGIN
    -- Деактивируем истекшие сессии
    UPDATE security.user_sessions
    SET is_active = FALSE,
        revoked_at = NOW(),
        revoked_reason = 'Session expired'
    WHERE is_active = TRUE
        AND expires_at < NOW();

    GET DIAGNOSTICS cleaned_count = ROW_COUNT;

    RETURN cleaned_count;
END;
$$ LANGUAGE plpgsql;

-- Функция для архивирования старых audit-логов (для соблюдения политики хранения)
CREATE OR REPLACE FUNCTION security.archive_old_audit_logs(
    p_retention_days INTEGER DEFAULT 2555 -- ~7 лет по умолчанию
) RETURNS INTEGER AS $$
DECLARE
    archive_count INTEGER;
BEGIN
    -- Создаем архивную таблицу если не существует
    CREATE TABLE IF NOT EXISTS security.audit_logs_archive (
        LIKE security.audit_logs INCLUDING ALL
    );

    -- Перемещаем старые записи в архив
    WITH moved_logs AS (
        DELETE FROM security.audit_logs
        WHERE timestamp < NOW() - INTERVAL '1 day' * p_retention_days
        RETURNING *
    )
    INSERT INTO security.audit_logs_archive
    SELECT * FROM moved_logs;

    GET DIAGNOSTICS archive_count = ROW_COUNT;

    RETURN archive_count;
END;
$$ LANGUAGE plpgsql;

-- Комментарии к таблицам для документации
COMMENT ON TABLE security.audit_logs IS 'Неизменяемая таблица для хранения всех действий пользователей в системе';
COMMENT ON COLUMN security.audit_logs.integrity_hash IS 'SHA256 хеш для проверки целостности записи';
COMMENT ON TABLE security.user_sessions IS 'Активные пользовательские сессии с поддержкой отзыва токенов';

-- Предоставляем права на чтение audit-логов только администраторам
-- (права должны настраиваться отдельно в зависимости от развертывания)
