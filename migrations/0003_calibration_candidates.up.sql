-- Face captures from this household's cameras, kept to calibrate raw similarity into a match
-- probability. Unlabeled and rejected rows (and their crops) are purged after
-- calibration_capture.unlabeled_retention_days; labeled rows are the dataset and stay until an
-- operator deletes them.
CREATE TABLE IF NOT EXISTS calibration_candidates (
    candidate_id          UUID PRIMARY KEY,
    captured_at           TIMESTAMPTZ NOT NULL,
    camera_id             TEXT NOT NULL,
    source                TEXT NOT NULL CHECK (source IN ('cts_identify', 'enrollment', 'external')),
    crop_object_key       TEXT NOT NULL,
    embedding             vector(512) NOT NULL,
    arcface_model_version TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    quality_version       TEXT NOT NULL,
    quality_score         DOUBLE PRECISION NOT NULL CHECK (quality_score BETWEEN 0 AND 1),
    quality_components    JSONB NOT NULL CHECK (jsonb_typeof(quality_components) = 'object'),
    capture_profile       JSONB NOT NULL CHECK (jsonb_typeof(capture_profile) = 'object'),
    interocular_px        DOUBLE PRECISION,
    yaw_deg               DOUBLE PRECISION NOT NULL,
    pitch_deg             DOUBLE PRECISION NOT NULL,
    nearest_person_id     TEXT,
    nearest_similarity    DOUBLE PRECISION NOT NULL,
    label_state           TEXT NOT NULL DEFAULT 'unlabeled'
        CHECK (label_state IN ('unlabeled', 'labeled', 'rejected')),
    label_identity_id     TEXT,
    label_kind            TEXT CHECK (label_kind IN ('member', 'not_enrolled', 'not_a_face')),
    label_source          TEXT CHECK (label_source IN ('operator_reviewed', 'enrollment_capture',
                                                       'external_benchmark', 'sensor_attested')),
    labeled_by            TEXT,
    labeled_at            TIMESTAMPTZ,
    -- A labeled row names its kind, source, labeler and time; only a member label names an identity.
    CONSTRAINT calibration_candidates_label_complete CHECK (
        (label_state = 'labeled') = (label_kind IS NOT NULL)
        AND (label_kind IS NULL OR (label_source IS NOT NULL AND labeled_by IS NOT NULL
                                    AND labeled_at IS NOT NULL))
        AND (COALESCE(label_kind = 'member', false) = (label_identity_id IS NOT NULL))
    )
);
CREATE INDEX IF NOT EXISTS idx_cal_cand_state_time
    ON calibration_candidates (label_state, captured_at DESC, candidate_id DESC);
