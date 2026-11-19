-- Multi-prototype enrollment: per-pose-bin prototypes, embedding provenance, and gallery revision.
--
-- Each enrolled member can hold several pose-binned prototypes (frontal, left, right, down) built
-- from enrollment photos, visitor-cluster namings, or operator-promoted captures. Exact cosine
-- search over face_prototypes is used (household galleries have a few hundred rows, so an ANN
-- index adds approximation risk without latency benefit).
--
-- members.gallery_revision increments whenever a member's gallery changes (enrollment update,
-- promotion, or prototype removal) so calibration artifacts fitted on an older revision fail
-- closed for that identity until refitted.

ALTER TABLE members
    ADD COLUMN IF NOT EXISTS gallery_revision INTEGER NOT NULL DEFAULT 1;

ALTER TABLE members DROP CONSTRAINT IF EXISTS members_gallery_revision_check;
ALTER TABLE members ADD CONSTRAINT members_gallery_revision_check
    CHECK (gallery_revision >= 1);

ALTER TABLE embeddings
    ADD COLUMN IF NOT EXISTS pose_bin TEXT NULL,
    ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'enrollment_photo',
    ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION NULL,
    ADD COLUMN IF NOT EXISTS quality_version TEXT NULL,
    ADD COLUMN IF NOT EXISTS source_candidate_id UUID NULL,
    ADD COLUMN IF NOT EXISTS approved_by TEXT NULL,
    ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS crop_object_key TEXT NULL;

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_pose_bin_check;
ALTER TABLE embeddings ADD CONSTRAINT embeddings_pose_bin_check
    CHECK (pose_bin IS NULL OR pose_bin IN ('frontal', 'left', 'right', 'down'));

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_source_check;
ALTER TABLE embeddings ADD CONSTRAINT embeddings_source_check
    CHECK (source IN ('enrollment_photo', 'deployment_capture', 'visitor_naming',
                      'wearable_exemplar'));

ALTER TABLE embeddings DROP CONSTRAINT IF EXISTS embeddings_quality_pair_check;
ALTER TABLE embeddings ADD CONSTRAINT embeddings_quality_pair_check
    CHECK ((quality_score IS NULL) = (quality_version IS NULL));

CREATE TABLE IF NOT EXISTS face_prototypes (
    prototype_id        UUID PRIMARY KEY,
    person_id           TEXT NOT NULL REFERENCES members(person_id) ON DELETE CASCADE,
    pose_bin            TEXT NOT NULL CHECK (pose_bin IN ('frontal', 'left', 'right', 'down')),
    embedding           vector(512) NOT NULL,
    member_count        INTEGER NOT NULL CHECK (member_count >= 1),
    source              TEXT NOT NULL CHECK (source IN ('enrollment_photo', 'deployment_capture',
                                                        'visitor_naming', 'wearable_exemplar')),
    quality_score       DOUBLE PRECISION NULL,
    quality_version     TEXT NULL,
    crop_object_key     TEXT NULL,
    source_embedding_id BIGINT NULL,
    approved_by         TEXT NULL,
    approved_at         TIMESTAMPTZ NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT face_prototypes_quality_pair_check
        CHECK ((quality_score IS NULL) = (quality_version IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_face_prototypes_person ON face_prototypes (person_id);
