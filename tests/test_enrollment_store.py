"""Tests for EnrollmentStore with TimescaleDB + pgvector."""

import os

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.services.enrollment_store import EnrollmentStore, ensure_schema

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


@pytest_asyncio.fixture
async def pool():
    """Create a test pool and ensure schema, skipping if DB is unavailable."""
    if not await _can_connect():
        pytest.skip("Database not available")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await ensure_schema(pool)
    yield pool
    # Clean up test data
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM guest_visits")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


@pytest_asyncio.fixture
async def store(pool):
    """Create an EnrollmentStore backed by the test pool."""

    class _FakeFaceEngine:
        """Fake FaceEngine that returns synthetic 512-dim embeddings."""

        def detect_faces(self, image: np.ndarray):
            rng = np.random.RandomState(42)
            embedding = rng.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            from app.services.face_models import DetectedFace

            return [
                DetectedFace(
                    bbox=[10.0, 20.0, 100.0, 120.0],
                    embedding=embedding,
                    det_score=0.95,
                )
            ]

    engine = _FakeFaceEngine()
    return EnrollmentStore(pool, engine)


class TestEnrollmentStore:
    async def test_enroll_new_member(self, store):
        """Enroll a new member and verify it appears in listings."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await store.enroll("test-1", "Alice", [img])

        assert result.status == "enrolled"
        assert result.person_id == "test-1"
        assert result.name == "Alice"
        assert result.embedding_count == 1
        assert result.failed_images == []

    async def test_enroll_update_existing(self, store):
        """Adding more images to an existing member updates the centroid."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        await store.enroll("test-2", "Bob", [img])
        result = await store.enroll("test-2", "Bob", [img])

        assert result.status == "updated"
        assert result.embedding_count == 2

    async def test_enroll_no_face_detected(self, store):
        """Images where no face is detected should fail gracefully."""

        class _NoFaceEngine:
            def detect_faces(self, image):
                return []

        store_no_face = EnrollmentStore(store._pool, _NoFaceEngine())

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await store_no_face.enroll("test-noface", "Ghost", [img])

        assert result.status == "failed"
        assert result.embedding_count == 0

    async def test_list_members(self, store):
        """List members returns enrolled persons."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        await store.enroll("test-3", "Charlie", [img])
        await store.enroll("test-4", "Diana", [img])

        members = await store.list_members()

        assert len(members) >= 2
        names = {m.name for m in members}
        assert "Charlie" in names
        assert "Diana" in names

    async def test_get_member(self, store):
        """Get a single member by ID."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        await store.enroll("test-5", "Eve", [img])

        member = await store.get_member("test-5")
        assert member is not None
        assert member.name == "Eve"
        assert member.person_id == "test-5"

    async def test_get_member_not_found(self, store):
        """Getting a non-existent member returns None."""
        member = await store.get_member("nonexistent")
        assert member is None

    async def test_remove_member(self, store):
        """Removing a member deletes their data."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        await store.enroll("test-6", "Frank", [img])
        assert await store.get_member("test-6") is not None

        deleted = await store.remove_member("test-6")
        assert deleted is True
        assert await store.get_member("test-6") is None

    async def test_remove_nonexistent_member(self, store):
        """Removing a non-existent member returns False."""
        deleted = await store.remove_member("no-one")
        assert deleted is False

    async def test_member_count(self, store):
        """Member count reflects enrolled members."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        initial = await store.member_count()
        await store.enroll("test-7", "Grace", [img])
        after_enroll = await store.member_count()

        assert after_enroll == initial + 1

    async def test_identify_known_person(self, store):
        """Identifying a face against a known member returns positive ID."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Enroll first, then use the same image for identification
        result = await store.enroll("test-8", "Hank", [img])
        assert result.status == "enrolled"

        # Re-detect and identify
        faces = store._engine.detect_faces(img)
        identity = await store.identify(faces[0].embedding)

        assert identity.person_id == "test-8"
        assert identity.name == "Hank"
        assert identity.confidence > 0.9  # Same embedding should be near-identical

    async def test_identify_unknown_person(self, store):
        """A face that doesn't match any centroid is classified as unknown."""
        # Use a random embedding that won't match anything
        rng = np.random.RandomState(99)
        unknown_emb = rng.randn(512).astype(np.float32)
        unknown_emb = unknown_emb / np.linalg.norm(unknown_emb)

        identity = await store.identify(unknown_emb)

        assert identity.person_id == "unknown"
        assert identity.name == "Guest"

    async def test_identify_empty_gallery(self, store):
        """Identifying against an empty gallery returns unknown."""
        rng = np.random.RandomState(99)
        emb = rng.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        identity = await store.identify(emb)

        assert identity.person_id == "unknown"

    async def test_identify_all(self, store):
        """Batch identify returns results for all faces with bboxes."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        await store.enroll("test-9", "Ivy", [img])

        faces = store._engine.detect_faces(img)
        faces[0].bbox = [10.0, 20.0, 100.0, 120.0]

        results = await store.identify_all(faces)

        assert len(results) == 1
        assert results[0].bbox == [10.0, 20.0, 100.0, 120.0]
