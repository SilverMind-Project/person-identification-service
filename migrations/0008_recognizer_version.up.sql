-- Scope every face-embedding table by recognizer_version so parallel galleries for a new
-- recognizer profile can be built and validated without modifying the active gallery.
--
-- The existing vector(512) columns serve all 512-dimensional profiles (for example buffalo_l and
-- AdaFace IR-50/IR-101). Introducing a recognizer with a different embedding dimension requires
-- profile-scoped tables created by a dedicated migration.

ALTER TABLE embeddings
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

CREATE INDEX IF NOT EXISTS idx_embeddings_person_recognizer
    ON embeddings (person_id, recognizer_version, created_at, id);

ALTER TABLE centroids
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'centroids'::regclass
          AND contype = 'p'
          AND pg_get_constraintdef(oid) = 'PRIMARY KEY (person_id)'
    ) THEN
        ALTER TABLE centroids DROP CONSTRAINT centroids_pkey;
        ALTER TABLE centroids ADD CONSTRAINT centroids_pkey
            PRIMARY KEY (person_id, recognizer_version);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_centroids_recognizer
    ON centroids (recognizer_version);

ALTER TABLE face_prototypes
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

CREATE INDEX IF NOT EXISTS idx_face_prototypes_person_recognizer
    ON face_prototypes (person_id, recognizer_version);

ALTER TABLE calibration_candidates
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

UPDATE calibration_candidates
SET recognizer_version = arcface_model_version
WHERE recognizer_version <> arcface_model_version;

CREATE INDEX IF NOT EXISTS idx_cal_cand_recognizer_state
    ON calibration_candidates (recognizer_version, label_state, captured_at DESC);

ALTER TABLE visitor_clusters
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

CREATE INDEX IF NOT EXISTS idx_visitor_clusters_recognizer_status
    ON visitor_clusters (recognizer_version, status);

ALTER TABLE visitor_sightings
    ADD COLUMN IF NOT EXISTS recognizer_version TEXT NOT NULL DEFAULT 'buffalo_l';

CREATE INDEX IF NOT EXISTS idx_visitor_sightings_recognizer_cluster
    ON visitor_sightings (recognizer_version, cluster_id, seen_at DESC);
