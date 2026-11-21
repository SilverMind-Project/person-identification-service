-- Refuse rollback while any table stores rows for a recognizer other than buffalo_l so parallel
-- galleries are never silently dropped and restoring centroids PRIMARY KEY (person_id) cannot fail.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM embeddings WHERE recognizer_version <> 'buffalo_l')
       OR EXISTS (SELECT 1 FROM centroids WHERE recognizer_version <> 'buffalo_l')
       OR EXISTS (SELECT 1 FROM face_prototypes WHERE recognizer_version <> 'buffalo_l')
       OR EXISTS (SELECT 1 FROM calibration_candidates WHERE recognizer_version <> 'buffalo_l')
       OR EXISTS (SELECT 1 FROM visitor_clusters WHERE recognizer_version <> 'buffalo_l')
       OR EXISTS (SELECT 1 FROM visitor_sightings WHERE recognizer_version <> 'buffalo_l') THEN
        RAISE EXCEPTION 'Cannot roll back 0008_recognizer_version while non-buffalo_l rows exist';
    END IF;
END $$;

DROP INDEX IF EXISTS idx_visitor_sightings_recognizer_cluster;
ALTER TABLE visitor_sightings
    DROP COLUMN IF EXISTS recognizer_version;

DROP INDEX IF EXISTS idx_visitor_clusters_recognizer_status;
ALTER TABLE visitor_clusters
    DROP COLUMN IF EXISTS recognizer_version;

DROP INDEX IF EXISTS idx_cal_cand_recognizer_state;
ALTER TABLE calibration_candidates
    DROP COLUMN IF EXISTS recognizer_version;

DROP INDEX IF EXISTS idx_face_prototypes_person_recognizer;
ALTER TABLE face_prototypes
    DROP COLUMN IF EXISTS recognizer_version;

DROP INDEX IF EXISTS idx_centroids_recognizer;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'centroids'::regclass
          AND contype = 'p'
          AND pg_get_constraintdef(oid) = 'PRIMARY KEY (person_id, recognizer_version)'
    ) THEN
        ALTER TABLE centroids DROP CONSTRAINT centroids_pkey;
        ALTER TABLE centroids ADD CONSTRAINT centroids_pkey PRIMARY KEY (person_id);
    END IF;
END $$;
ALTER TABLE centroids
    DROP COLUMN IF EXISTS recognizer_version;

DROP INDEX IF EXISTS idx_embeddings_person_recognizer;
ALTER TABLE embeddings
    DROP COLUMN IF EXISTS recognizer_version;
