-- Migration 007: create stream_stats table

CREATE TABLE IF NOT EXISTS stream_stats (
    id UUID PRIMARY KEY,
    equipment_id UUID NULL REFERENCES equipment(id) ON DELETE CASCADE,
    feature_id UUID NULL REFERENCES features(id) ON DELETE CASCADE,
    raw_id UUID NULL REFERENCES raw_signals(id) ON DELETE CASCADE,
    detector VARCHAR(50) NOT NULL,
    metric VARCHAR(50) NOT NULL,
    value NUMERIC(14,6) NOT NULL,
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_stream_stats_equipment_created ON stream_stats(equipment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_stream_stats_detector_metric ON stream_stats(detector, metric);
CREATE INDEX IF NOT EXISTS idx_stream_stats_drift ON stream_stats(drift_detected);
