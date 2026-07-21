"""Enrollment store: manages face embedding gallery in TimescaleDB with pgvector + DiskANN."""

from __future__ import annotations

import logging
from typing import Protocol

import asyncpg
import numpy as np

from app import config
from app.models.enrollment import EnrollResult, MemberInfo
from app.services.centroid import insert_embeddings, recompute_member_centroid
from app.services.face_models import DetectedFace, IdentifyResult

logger = logging.getLogger(__name__)


class FaceEngineProtocol(Protocol):
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]: ...


class EnrollmentStore:
    """Manages a gallery of enrolled household members with face embeddings.

    Embeddings are stored in TimescaleDB (pgvector). Identification uses
    pre-computed centroids indexed with StreamingDiskANN via pgvectorscale.
    """

    def __init__(self, pool: asyncpg.Pool, face_engine: FaceEngineProtocol) -> None:
        self._pool = pool
        self._engine = face_engine
        self._threshold = float(config.get("recognition.threshold", 0.4))
        self._unknown_threshold = float(config.get("recognition.unknown_threshold", 0.25))

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
                faces = await self._engine.detect_faces(img)
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

            # Insert new embeddings and recompute the centroid (shared with
            # visitor-cluster naming; see app.services.centroid).
            await insert_embeddings(conn, person_id, embeddings)
            await recompute_member_centroid(conn, person_id)

            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM embeddings WHERE person_id = $1", person_id
            )
            total_count = row["cnt"]
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

        Always returns the best candidate and recognition state, even when
        similarity is below the recognition threshold.  The three states are:

        - ``recognized``:  similarity >= threshold (strong positive).
        - ``candidate``:   unknown_threshold <= similarity < threshold (grey zone).
        - ``unrecognized``: similarity < unknown_threshold (definitely unknown).

        For backward compatibility ``person_id`` stays ``"unknown"`` when below
        ``threshold``, but ``best_candidate_id`` always carries the nearest centroid.
        """
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
                return IdentifyResult(
                    person_id="unknown",
                    name="Guest",
                    confidence=0.0,
                    bbox=[],
                    best_candidate_id=None,
                    similarity=0.0,
                    recognition_state="unrecognized",
                )

            similarity = float(row["similarity"])
            best_id = str(row["person_id"])
            best_name = str(row["name"])

            if similarity >= self._threshold:
                recognition_state = "recognized"
                person_id = best_id
                name = best_name
                confidence = similarity
            elif similarity >= self._unknown_threshold:
                recognition_state = "candidate"
                person_id = "unknown"
                name = "Guest"
                confidence = similarity
            else:
                recognition_state = "unrecognized"
                person_id = "unknown"
                name = "Guest"
                confidence = max(0.0, similarity)

            return IdentifyResult(
                person_id=person_id,
                name=name,
                confidence=confidence,
                bbox=[],
                best_candidate_id=best_id,
                similarity=similarity,
                recognition_state=recognition_state,
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
            # Carry forward detection-level fields from the detected face.
            result.bbox = face.bbox
            result.yaw_deg = face.yaw_deg
            result.pitch_deg = face.pitch_deg
            result.roll_deg = face.roll_deg
            result.det_score = face.det_score
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
