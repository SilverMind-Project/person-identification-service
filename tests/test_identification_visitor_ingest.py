"""Tests that identify/identify-batch route unknown faces into VisitorStore.

Identity-continuity M06. Exercises the shared `_ingest_unknown_faces` helper
(app/routers/identification.py) directly, since it is called identically by
both endpoints and the DB-level effect is what matters.
"""

from __future__ import annotations

import os

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.routers.identification import _ingest_unknown_faces
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.visitor_store import VisitorStore

TEST_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://pid_user:change-me-pid-password@localhost:5432/person_identification",
)


async def _can_connect() -> bool:
    try:
        conn = await asyncpg.connect(TEST_DSN, timeout=5)
        await conn.close()
        return True
    except Exception:
        return False


class _FakeMinioClient:
    def __init__(self):
        self.uploads: dict[str, bytes] = {}
        self.bucket = "test-bucket"

    def ensure_bucket(self) -> None:
        pass

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "image/jpeg") -> str:
        self.uploads[object_name] = data
        return object_name

    def generate_presigned_url(self, object_name: str, expiration: int = 3600) -> str:
        return f"http://fake-minio/{self.bucket}/{object_name}"

    def delete_object(self, object_name: str) -> None:
        self.uploads.pop(object_name, None)


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _face(embedding: np.ndarray) -> DetectedFace:
    return DetectedFace(bbox=[10.0, 20.0, 110.0, 140.0], embedding=embedding, det_score=0.95)


def _identity(person_id: str, similarity: float = 0.1) -> IdentifyResult:
    return IdentifyResult(
        person_id=person_id,
        name="Guest" if person_id == "unknown" else "Alice",
        confidence=similarity,
        bbox=[10.0, 20.0, 110.0, 140.0],
        best_candidate_id=None,
        similarity=similarity,
        recognition_state="unrecognized" if person_id == "unknown" else "recognized",
    )


@pytest_asyncio.fixture
async def pool():
    if not await _can_connect():
        pytest.skip("Database not available")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM visitor_sightings")
        await conn.execute("DELETE FROM visitor_clusters")
        await conn.execute("DELETE FROM guest_visits")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


async def _cluster_count(pool) -> int:
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT count(*) AS cnt FROM visitor_clusters")
        return row["cnt"]


class TestIdentificationVisitorIngest:
    async def test_unknown_face_records_sighting_when_enabled(self, pool):
        store = VisitorStore(pool, _FakeMinioClient())
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        await _ingest_unknown_faces(
            store, image, [_face(_embedding(1))], [_identity("unknown")]
        )

        assert await _cluster_count(pool) == 1

    async def test_unknown_face_not_recorded_when_disabled(self, pool):
        store = VisitorStore(pool, _FakeMinioClient())
        store._clustering_enabled = False
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        await _ingest_unknown_faces(
            store, image, [_face(_embedding(2))], [_identity("unknown")]
        )

        assert await _cluster_count(pool) == 0

    async def test_recognized_face_never_touches_visitor_store(self, pool):
        store = VisitorStore(pool, _FakeMinioClient())
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        await _ingest_unknown_faces(
            store, image, [_face(_embedding(3))], [_identity("alice", similarity=0.95)]
        )

        assert await _cluster_count(pool) == 0
