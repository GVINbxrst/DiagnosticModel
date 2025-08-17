-- =============================================================================
-- DiagMod Database Schema Migration 004 - Forecasts Table
-- Добавляет таблицу прогнозов временных рядов (RMS и др.)
-- Зависимости: 001_initial.sql, 002_indexes.sql
-- =============================================================================

SET search_path TO diagmod, public;

-- Создание таблицы прогнозов
CREATE TABLE IF NOT EXISTS forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_id UUID REFERENCES raw_signals(id) ON DELETE SET NULL,
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    horizon INTEGER NOT NULL DEFAULT 24,                 -- Кол-во шагов прогноза
    method VARCHAR(50) NOT NULL,                         -- Метод / модель (Prophet, FallbackMA, etc.)
    forecast_data JSONB NOT NULL,                        -- Полный объект прогноза
    probability_over_threshold NUMERIC(5,4),             -- Вероятность превышения порога
    model_version VARCHAR(20),                           -- Версия модели прогноза
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Индекс для частых выборок по оборудованию и времени
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_forecasts_equipment_created
    ON forecasts (equipment_id, created_at DESC);

-- Примечание: миграция не удаляет объекты при повторном запуске (idempotent по созданию)
