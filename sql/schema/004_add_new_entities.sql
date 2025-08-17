-- ============================================================================
-- Additional schema objects for DiagMod (post-initial)                      
-- Adds new analytical / ML tables introduced after initial schema:         
--  * defect_catalog, cluster_labels                                        
--  * forecasts (time series predictions + risk)                            
--  * anomaly_scores (stream/incremental anomaly detection)                 
--  * stream_stats (drift / streaming metrics)                              
--  * hourly_feature_summary (hourly RMS aggregates)                         
--  * schema evolution for features (cluster_id, severity_score columns)    
-- This file is idempotent (safe to re-run).                                
-- ============================================================================
SET search_path TO diagmod, public;

-- -------- Helper: add column if missing ------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'diagmod' AND table_name='features' AND column_name='cluster_id'
    ) THEN
        ALTER TABLE features ADD COLUMN cluster_id INTEGER;
        CREATE INDEX IF NOT EXISTS idx_features_cluster_id ON features(cluster_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'diagmod' AND table_name='features' AND column_name='severity_score'
    ) THEN
        ALTER TABLE features ADD COLUMN severity_score NUMERIC(5,4);
    END IF;
END$$;

-- -------- defect_catalog ----------------------------------------------------
CREATE TABLE IF NOT EXISTS defect_catalog (
    defect_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    severity_scale VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- -------- cluster_labels ----------------------------------------------------
CREATE TABLE IF NOT EXISTS cluster_labels (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER UNIQUE NOT NULL,
    defect_id VARCHAR(20) NOT NULL REFERENCES defect_catalog(defect_id) ON DELETE CASCADE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_cluster_labels_cluster_id ON cluster_labels(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_labels_defect_id ON cluster_labels(defect_id);

-- -------- forecasts ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    raw_id UUID REFERENCES raw_signals(id) ON DELETE SET NULL,
    equipment_id UUID REFERENCES equipment(id) ON DELETE CASCADE,
    horizon INTEGER NOT NULL DEFAULT 24,
    method VARCHAR(50) NOT NULL DEFAULT 'simple_trend',
    forecast_data JSONB NOT NULL DEFAULT '{}',
    probability_over_threshold NUMERIC(5,4),
    model_version VARCHAR(20),
    risk_score NUMERIC(5,4),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_forecasts_equipment_created ON forecasts(equipment_id, created_at);

-- -------- anomaly_scores ----------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_id UUID NOT NULL REFERENCES features(id) ON DELETE CASCADE,
    raw_id UUID NOT NULL REFERENCES raw_signals(id) ON DELETE CASCADE,
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    model_type VARCHAR(64) NOT NULL,               -- isolation_forest | half_space_trees | sliding_if
    model_version VARCHAR(32),
    score NUMERIC(12,6) NOT NULL,
    is_anomaly BOOLEAN NOT NULL DEFAULT false,
    threshold NUMERIC(12,6),
    probability NUMERIC(5,4) NOT NULL,             -- duplicate logical field (probability of anomaly)
    anomaly_detected BOOLEAN NOT NULL DEFAULT false,
    confidence NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    predicted_severity defect_severity,
    confidence_score NUMERIC(5,4),
    model_name VARCHAR(100) NOT NULL DEFAULT 'stream_model',
    prediction_details JSONB DEFAULT '{}',
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_equipment_created ON anomaly_scores(equipment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_anomaly_scores_is_anomaly ON anomaly_scores(is_anomaly);
CREATE INDEX IF NOT EXISTS idx_predictions_feature_id ON anomaly_scores(feature_id);
CREATE INDEX IF NOT EXISTS idx_predictions_probability ON anomaly_scores(probability);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON anomaly_scores(model_name, model_version);

-- -------- stream_stats ------------------------------------------------------
CREATE TABLE IF NOT EXISTS stream_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id UUID REFERENCES equipment(id) ON DELETE CASCADE,
    feature_id UUID REFERENCES features(id) ON DELETE CASCADE,
    raw_id UUID REFERENCES raw_signals(id) ON DELETE CASCADE,
    detector VARCHAR(50) NOT NULL,                -- adwin | page_hinkley
    metric VARCHAR(50) NOT NULL,                  -- e.g. rms_a
    value NUMERIC(14,6) NOT NULL,
    drift_detected BOOLEAN NOT NULL DEFAULT false,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_stream_stats_equipment_created ON stream_stats(equipment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_stream_stats_detector_metric ON stream_stats(detector, metric);

-- -------- hourly_feature_summary -------------------------------------------
CREATE TABLE IF NOT EXISTS hourly_feature_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    equipment_id UUID NOT NULL REFERENCES equipment(id) ON DELETE CASCADE,
    hour_start TIMESTAMPTZ NOT NULL,
    rms_mean NUMERIC(12,6),
    rms_max NUMERIC(12,6),
    rms_min NUMERIC(12,6),
    samples INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_hourly_feature_equipment_hour UNIQUE (equipment_id, hour_start)
);
CREATE INDEX IF NOT EXISTS idx_hourly_feature_equipment_hour ON hourly_feature_summary(equipment_id, hour_start);

-- -------- End --------------------------------------------------------------
