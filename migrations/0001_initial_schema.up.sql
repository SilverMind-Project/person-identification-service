-- Person Identification Service — consolidated baseline schema
--
-- Squash of the former chain (0001_initial_schema, 0002_visitor_clusters) into
-- the final state. Both migrations were purely additive: no table, column, or
-- index created by an earlier migration was dropped or altered by a later one,
-- so this baseline is a faithful concatenation with nothing excluded.
--
-- TimescaleDB hypertables + pgvector DiskANN index for face recognition.
CREATE EXTENSION IF NOT EXISTS vector CASCADE;
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Member metadata
CREATE TABLE IF NOT EXISTS members (
    person_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Raw embeddings hypertable
CREATE TABLE IF NOT EXISTS embeddings (
    id          BIGINT GENERATED ALWAYS AS IDENTITY,
    person_id   TEXT NOT NULL REFERENCES members(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
SELECT create_hypertable('embeddings', 'created_at', if_not_exists => TRUE);

-- Pre-computed centroids with DiskANN index
CREATE TABLE IF NOT EXISTS centroids (
    person_id   TEXT PRIMARY KEY REFERENCES members(person_id) ON DELETE CASCADE,
    centroid    vector(512) NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_centroids_diskann
    ON centroids USING diskann (centroid vector_cosine_ops);

-- Guest visit log (hypertable)
CREATE TABLE IF NOT EXISTS guest_visits (
    id            BIGINT GENERATED ALWAYS AS IDENTITY,
    guest_count   INT NOT NULL,
    object_name   TEXT NOT NULL,
    frame_index   INT NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
SELECT create_hypertable('guest_visits', 'created_at', if_not_exists => TRUE);

-- Visitor clustering (identity-continuity M06)
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
