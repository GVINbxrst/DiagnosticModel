-- Migration 006: add cluster_id to features and create cluster_labels table

-- For Postgres
ALTER TABLE features ADD COLUMN IF NOT EXISTS cluster_id INTEGER;
CREATE INDEX IF NOT EXISTS idx_features_cluster_id ON features(cluster_id);

CREATE TABLE IF NOT EXISTS cluster_labels (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER UNIQUE NOT NULL,
    defect_label VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_cluster_labels_cluster_id ON cluster_labels(cluster_id);
