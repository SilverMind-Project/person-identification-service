-- Person Identification Service — initial schema
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
