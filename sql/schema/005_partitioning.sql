-- Почасовое/дневное партиционирование крупных таблиц raw_signals и features
-- Идempotent: создаёт партиционированные родительские таблицы если ещё не преобразованы.

-- RAW_SIGNALS PARTITIONING (by month)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_inherits i ON c.oid=i.inhrelid
        JOIN pg_class p ON i.inhparent = p.oid
        WHERE p.relname='raw_signals') THEN
        -- Переименование оригинальной таблицы
        ALTER TABLE IF EXISTS raw_signals RENAME TO raw_signals_base;
        -- Создаём родительскую партиционированную таблицу
        CREATE TABLE raw_signals (
            LIKE raw_signals_base INCLUDING ALL
        ) PARTITION BY RANGE (recorded_at);
        -- Перенос индексов/констрейнтов (LIKЕ ... INCLUDING ALL сделало копию)
        INSERT INTO raw_signals SELECT * FROM raw_signals_base;
        DROP TABLE raw_signals_base;
    END IF;
END$$;

-- FEATURES PARTITIONING (by month)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_inherits i ON c.oid=i.inhrelid
        JOIN pg_class p ON i.inhparent = p.oid
        WHERE p.relname='features') THEN
        ALTER TABLE IF EXISTS features RENAME TO features_base;
        CREATE TABLE features (
            LIKE features_base INCLUDING ALL
        ) PARTITION BY RANGE (window_start);
        INSERT INTO features SELECT * FROM features_base;
        DROP TABLE features_base;
    END IF;
END$$;

-- Helper function to create monthly partitions dynamically
CREATE OR REPLACE FUNCTION ensure_month_partition(base_table text, part_col text, from_date date, to_date date)
RETURNS VOID AS $$
DECLARE
    d date;
    part_name text;
    start_ts timestamptz;
    end_ts timestamptz;
    ddl text;
BEGIN
    d := date_trunc('month', from_date);
    WHILE d <= to_date LOOP
        part_name := base_table || '_' || to_char(d, 'YYYY_MM');
        start_ts := d;
        end_ts := (d + INTERVAL '1 month');
        -- Проверяем существование
        IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = part_name) THEN
            ddl := format('CREATE TABLE %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L);',
                          part_name, base_table, start_ts, end_ts);
            EXECUTE ddl;
        END IF;
        d := d + INTERVAL '1 month';
    END LOOP;
END;$$ LANGUAGE plpgsql;

-- Создаём партиции на 12 месяцев вперёд и 12 назад
SELECT ensure_month_partition('raw_signals','recorded_at', now() - INTERVAL '12 months', now() + INTERVAL '12 months');
SELECT ensure_month_partition('features','window_start', now() - INTERVAL '12 months', now() + INTERVAL '12 months');

-- Индексы на родительских таблицах не наследуются автоматически для декларативного партиционирования.
-- Создадим необходимые индексы если отсутствуют.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='raw_signals' AND indexname='idx_raw_signals_equipment_recorded') THEN
        CREATE INDEX idx_raw_signals_equipment_recorded ON raw_signals (equipment_id, recorded_at);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename='features' AND indexname='idx_features_raw_window') THEN
        CREATE INDEX idx_features_raw_window ON features (raw_id, window_start);
    END IF;
END$$;
