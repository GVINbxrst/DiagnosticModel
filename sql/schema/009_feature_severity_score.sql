-- Migration 009: add severity_score to features
ALTER TABLE features ADD COLUMN IF NOT EXISTS severity_score NUMERIC(5,4);
