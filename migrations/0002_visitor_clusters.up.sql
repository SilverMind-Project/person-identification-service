-- Person Identification Service — visitor clustering (identity-continuity M06)
-- Persists unmatched face embeddings, clusters them across visits, and tracks
-- review state for naming a recurring visitor into a household member.
CREATE TABLE IF NOT EXISTS visitor_clusters (
    cluster_id      UUID PRIMARY KEY,
    status          TEXT NOT NULL DEFAULT 'candidate',
        -- candidate | surfaced | named | dismissed
    display_hint    TEXT NULL,          -- operator-visible label before naming, optional
    named_person_id TEXT NULL REFERENCES members(person_id),
    centroid        vector(512) NOT NULL,
    sighting_count  INTEGER NOT NULL DEFAULT 0,
    distinct_days   INTEGER NOT NULL DEFAULT 0,
    first_seen_at   TIMESTAMPTZ NOT NULL,
    last_seen_at    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT visitor_clusters_status_check
        CHECK (status IN ('candidate', 'surfaced', 'named', 'dismissed'))
);

CREATE TABLE IF NOT EXISTS visitor_sightings (
    id            BIGINT GENERATED ALWAYS AS IDENTITY,
    cluster_id    UUID NOT NULL REFERENCES visitor_clusters(cluster_id) ON DELETE CASCADE,
    embedding     vector(512) NOT NULL,
    quality       REAL NOT NULL DEFAULT 0.0,     -- detector score or blur proxy
    crop_object   TEXT NULL,                     -- MinIO key of the face crop
    seen_at       TIMESTAMPTZ NOT NULL,
    source        TEXT NOT NULL DEFAULT 'identify'
);
SELECT create_hypertable('visitor_sightings', 'seen_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_visitor_sightings_cluster_seen
    ON visitor_sightings (cluster_id, seen_at DESC);
