DROP INDEX IF EXISTS idx_face_prototypes_person;
DROP TABLE IF EXISTS face_prototypes;

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_quality_pair_check;
ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_source_check;
ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_pose_bin_check;

ALTER TABLE embeddings
    DROP COLUMN IF EXISTS crop_object_key,
    DROP COLUMN IF EXISTS approved_at,
    DROP COLUMN IF EXISTS approved_by,
    DROP COLUMN IF EXISTS source_candidate_id,
    DROP COLUMN IF EXISTS quality_version,
    DROP COLUMN IF EXISTS quality_score,
    DROP COLUMN IF EXISTS source,
    DROP COLUMN IF EXISTS pose_bin;

ALTER TABLE members DROP CONSTRAINT IF EXISTS members_gallery_revision_check;
ALTER TABLE members DROP COLUMN IF EXISTS gallery_revision;
