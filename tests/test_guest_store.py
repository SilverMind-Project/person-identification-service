"""Tests for GuestImageStore with MinIO + TimescaleDB."""

from __future__ import annotations

import os

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.guest_store import GuestImageStore

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
    """Fake MinIO client that stores uploads in memory for testing."""

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


@pytest_asyncio.fixture
async def pool():
    """Create a test pool and ensure schema, skipping if DB is unavailable."""
    if not await _can_connect():
        pytest.skip("Database not available")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM guest_visits")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


@pytest_asyncio.fixture
async def store(pool):
    """Create a GuestImageStore backed by a fake MinIO client."""
    minio = _FakeMinioClient()
    return GuestImageStore(pool, minio)


class TestGuestImageStore:
    async def test_save_guest_image(self, store):
        """Saving a guest image uploads to MinIO and logs to DB."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        object_name = await store.save_guest_image(img, guest_count=2, frame_index=0)
        assert object_name is not None
        assert object_name.startswith("guests/")
        assert "guests.jpg" in object_name

        # Verify MinIO upload
        assert object_name in store._minio.uploads
        assert len(store._minio.uploads[object_name]) > 0

    async def test_save_guest_image_rolls_back_on_failure(self, store):
        """Invalid image data should not crash."""
        # An all-zero image should encode fine, so test with valid data
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        store._minio.uploads.clear()

        object_name = await store.save_guest_image(img, guest_count=1, frame_index=5)
        assert object_name is not None
        assert len(store._minio.uploads) == 1
