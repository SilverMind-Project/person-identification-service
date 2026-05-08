"""Enrollment store: manages face embedding gallery in TimescaleDB with pgvector + DiskANN."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import asyncpg
import numpy as np

from app import config
from app.models.enrollment import EnrollResult, MemberInfo

if TYPE_CHECKING:
    from app.services.face_engine import FaceEngine
    from app.services.face_models import DetectedFace, IdentifyResult

logger = logging.getLogger(__name__)

# DDL to ensure schema exists (idempotent: IF NOT EXISTS throughout)
_SCHEMA_DDL = """
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
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Run DDL to create tables, hypertables, and indexes if not present."""
    async with pool.acquire() as conn:
        await conn.execute(_SCHEMA_DDL)
    logger.info("Database schema ensured")


class EnrollmentStore:
    """Manages a gallery of enrolled household members with face embeddings.

    Embeddings are stored in TimescaleDB (pgvector). Identification uses
    pre-computed centroids indexed with StreamingDiskANN via pgvectorscale.
    """

    def __init__(self, pool: asyncpg.Pool, face_engine: FaceEngine) -> None:
        self._pool = pool
        self._engine = face_engine
        self._threshold = float(config.get("recognition.threshold", 0.4))

    async def enroll(self, person_id: str, name: str, images: list[np.ndarray]) -> EnrollResult:
        """Enroll a new person or add images to an existing member.

        Args:
            person_id: Unique identifier.
            name: Display name.
            images: List of BGR numpy arrays, each containing a face.

        Returns:
            EnrollResult with status and count of successful embeddings.
        """
        async with self._pool.acquire() as conn:
            # Check existing member
            existing = await conn.fetchrow(
                "SELECT person_id FROM members WHERE person_id = $1", person_id
            )
            existing_count = 0
            if existing:
                row = await conn.fetchrow(
                    "SELECT count(*) AS cnt FROM embeddings WHERE person_id = $1", person_id
                )
                existing_count = row["cnt"]

            # Detect faces (CPU/GPU-bound, run in threadpool)
            embeddings: list[np.ndarray] = []
            failed_indices: list[int] = []

            for idx, img in enumerate(images):
                faces = await asyncio.to_thread(self._engine.detect_faces, img)
                if not faces:
                    failed_indices.append(idx)
                    continue
                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                embeddings.append(best_face.embedding)

            if not embeddings:
                return EnrollResult(
                    person_id=person_id,
                    name=name,
                    embedding_count=existing_count,
                    status="failed",
                    failed_images=failed_indices,
                )

            # Insert or update member
            if existing:
                await conn.execute(
                    "UPDATE members SET name = $1 WHERE person_id = $2",
                    name,
                    person_id,
                )
                status = "updated"
            else:
                await conn.execute(
                    "INSERT INTO members (person_id, name) VALUES ($1, $2)",
                    person_id,
                    name,
                )
                status = "enrolled"

            # Insert new embeddings
            for emb in embeddings:
                await conn.execute(
                    "INSERT INTO embeddings (person_id, embedding) VALUES ($1, $2)",
                    person_id,
                    emb,
                )

            # Recompute centroid: load all embeddings for this person
            rows = await conn.fetch(
                "SELECT embedding FROM embeddings WHERE person_id = $1", person_id
            )
            all_embeddings = [r["embedding"] for r in rows]
            centroid = np.mean(all_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            # Upsert centroid
            await conn.execute(
                """INSERT INTO centroids (person_id, centroid, updated_at)
                   VALUES ($1, $2, now())
                   ON CONFLICT (person_id)
                   DO UPDATE SET centroid = excluded.centroid, updated_at = now()""",
                person_id,
                centroid,
            )

            total_count = len(all_embeddings)
            logger.info(
                "Enrolled person_id=%s name=%s embeddings=%d status=%s",
                person_id,
                name,
                total_count,
                status,
            )
            return EnrollResult(
                person_id=person_id,
                name=name,
                embedding_count=total_count,
                status=status,
                failed_images=failed_indices,
            )

    async def identify(self, embedding: np.ndarray) -> IdentifyResult:
        """Identify a detected face against the enrolled gallery using DiskANN index.

        Returns the best match or "unknown" if below threshold.
        """
        from app.services.face_models import IdentifyResult

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT m.person_id, m.name,
                          1.0 - (c.centroid <=> $1::vector) AS similarity
                   FROM centroids c
                   JOIN members m ON m.person_id = c.person_id
                   ORDER BY c.centroid <=> $1::vector
                   LIMIT 1""",
                embedding,
            )

            if row is None:
                return IdentifyResult(person_id="unknown", name="Guest", confidence=0.0, bbox=[])

            similarity = float(row["similarity"])
            if similarity < self._threshold:
                return IdentifyResult(
                    person_id="unknown",
                    name="Guest",
                    confidence=max(0.0, similarity),
                    bbox=[],
                )

            return IdentifyResult(
                person_id=row["person_id"],
                name=row["name"],
                confidence=similarity,
                bbox=[],
            )

    async def identify_all(self, faces: list[DetectedFace]) -> list[IdentifyResult]:
        """Identify all faces in a single frame.

        Each face gets its own nearest-centroid query so that the DiskANN
        index is used per embedding. For household-scale galleries this is
        fast; batch optimizations can be added later if needed.
        """
        results: list[IdentifyResult] = []
        for face in faces:
            result = await self.identify(face.embedding)
            # Carry forward the bbox from the detected face
            result.bbox = face.bbox
            results.append(result)
        return results

    async def list_members(self) -> list[MemberInfo]:
        """Return all enrolled members with embedding counts."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT m.person_id, m.name, m.created_at,
                          count(e.id)::int AS embedding_count
                   FROM members m
                   LEFT JOIN embeddings e ON e.person_id = m.person_id
                   GROUP BY m.person_id, m.name, m.created_at
                   ORDER BY m.name"""
            )

        return [
            MemberInfo(
                person_id=r["person_id"],
                name=r["name"],
                embedding_count=r["embedding_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def get_member(self, person_id: str) -> MemberInfo | None:
        """Get details of a specific enrolled member."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT m.person_id, m.name, m.created_at,
                          count(e.id)::int AS embedding_count
                   FROM members m
                   LEFT JOIN embeddings e ON e.person_id = m.person_id
                   WHERE m.person_id = $1
                   GROUP BY m.person_id, m.name, m.created_at""",
                person_id,
            )

        if not row:
            return None

        return MemberInfo(
            person_id=row["person_id"],
            name=row["name"],
            embedding_count=row["embedding_count"],
            created_at=row["created_at"],
        )

    async def remove_member(self, person_id: str) -> bool:
        """Remove a member and all their embeddings (CASCADE handles centroids too)."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM members WHERE person_id = $1", person_id)

        deleted = result != "DELETE 0"
        if deleted:
            logger.info("Removed member person_id=%s", person_id)
        return deleted

    async def member_count(self) -> int:
        """Return the number of enrolled members."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) AS cnt FROM members")
            return row["cnt"]
