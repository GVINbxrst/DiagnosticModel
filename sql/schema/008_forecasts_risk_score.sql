-- Migration 008: add risk_score column to forecasts
ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS risk_score NUMERIC(5,4);
