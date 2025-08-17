-- Миграция: таблица anomaly_scores для потоковых результатов аномалий
CREATE TABLE IF NOT EXISTS anomaly_scores (
    id UUID PRIMARY KEY,
    feature_id UUID NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    raw_id UUID NOT NULL REFERENCES raw_signals(id) ON DELETE CASCADE,
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    model_type VARCHAR(64) NOT NULL,
    model_version VARCHAR(32),
    score NUMERIC(12,6) NOT NULL,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE,
    threshold NUMERIC(12,6),
    meta JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_anomaly_scores_equipment_created ON anomaly_scores(equipment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_is_anomaly ON anomaly_scores(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_feature_id ON anomaly_scores(feature_id);
