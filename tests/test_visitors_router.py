"""Tests for the visitors router (identity-continuity M06/M07 contract).

Exercises the FastAPI route handler coroutines directly (this repo's tests
never spin up an HTTP layer; see test_enrollment_store.py precedent), with a
minimal fake Request exposing app.state.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from fastapi import HTTPException
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.models.visitor import NameClusterRequest
from app.routers import visitors
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


class _FakeRequest:
    def __init__(self, visitor_store: VisitorStore):
        self.app = SimpleNamespace(state=SimpleNamespace(visitor_store=visitor_store))


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _face(embedding: np.ndarray) -> DetectedFace:
    return DetectedFace(bbox=[10.0, 20.0, 110.0, 140.0], embedding=embedding, det_score=0.95)


def _unknown_identity() -> IdentifyResult:
    return IdentifyResult(
        person_id="unknown",
        name="Guest",
        confidence=0.1,
        bbox=[10.0, 20.0, 110.0, 140.0],
        best_candidate_id=None,
        similarity=0.1,
        recognition_state="unrecognized",
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


@pytest_asyncio.fixture
async def store(pool):
    return VisitorStore(pool, _FakeMinioClient())


@pytest_asyncio.fixture
async def request_(store):
    return _FakeRequest(store)


class TestListAndDetail:
    async def test_list_and_detail_shapes(self, store, request_):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(1)), _unknown_identity()
        )

        listing = await visitors.list_clusters(request_, status=None)
        assert listing.total == 1
        assert listing.clusters[0].cluster_id == cluster_id

        detail = await visitors.get_cluster(request_, cluster_id)
        assert detail.cluster_id == cluster_id
        assert len(detail.recent_sightings) == 1

    async def test_detail_404_for_missing_cluster(self, request_):
        with pytest.raises(HTTPException) as exc_info:
            await visitors.get_cluster(request_, "00000000-0000-0000-0000-000000000000")
        assert exc_info.value.status_code == 404


class TestNaming:
    async def test_name_happy_path(self, store, request_, pool):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(2)), _unknown_identity()
        )
        body = NameClusterRequest(person_id="nurse-priya", name="Nurse Priya")

        result = await visitors.name_cluster(request_, cluster_id, body)
        assert result.named_person_id == "nurse-priya"
        assert result.embedding_count == 1

        async with pool.acquire() as conn:
            member = await conn.fetchrow(
                "SELECT * FROM members WHERE person_id = $1", "nurse-priya"
            )
            centroid = await conn.fetchrow(
                "SELECT * FROM centroids WHERE person_id = $1", "nurse-priya"
            )
        assert member is not None
        assert centroid is not None

    async def test_name_conflict_existing_person_id(self, store, request_, pool):
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO members (person_id, name) VALUES ($1, $2)", "alice", "Alice"
            )
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(3)), _unknown_identity()
        )
        body = NameClusterRequest(person_id="alice", name="Someone Else")

        with pytest.raises(HTTPException) as exc_info:
            await visitors.name_cluster(request_, cluster_id, body)
        assert exc_info.value.status_code == 409

    async def test_name_rejects_invalid_slug(self, store, request_):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(4)), _unknown_identity()
        )
        body = NameClusterRequest(person_id="Nurse Priya!", name="Nurse Priya")

        with pytest.raises(HTTPException) as exc_info:
            await visitors.name_cluster(request_, cluster_id, body)
        assert exc_info.value.status_code == 400


class TestDismissAndMerge:
    async def test_dismiss(self, store, request_):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(5)), _unknown_identity()
        )
        result = await visitors.dismiss_cluster(request_, cluster_id)
        assert result["status"] == "dismissed"

    async def test_dismiss_missing_404(self, request_):
        with pytest.raises(HTTPException) as exc_info:
            await visitors.dismiss_cluster(request_, "00000000-0000-0000-0000-000000000000")
        assert exc_info.value.status_code == 404

    async def test_merge_recomputes_centroid(self, store, request_):
        emb_a = _embedding(6)
        rng = np.random.RandomState(7)
        emb_b = rng.randn(512).astype(np.float32)
        emb_b = emb_b / np.linalg.norm(emb_b)

        cluster_a = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(emb_a), _unknown_identity()
        )
        cluster_b = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(emb_b), _unknown_identity()
        )

        merged = await visitors.merge_clusters(request_, cluster_a, cluster_b)
        assert merged.sighting_count == 2


class TestDisabledFlag:
    async def test_disabled_list_empty_and_mutations_409(self, store, request_):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(8)), _unknown_identity()
        )
        store._clustering_enabled = False

        listing = await visitors.list_clusters(request_, status=None)
        assert listing.total == 0
        assert listing.clusters == []

        with pytest.raises(HTTPException) as exc_info:
            await visitors.dismiss_cluster(request_, cluster_id)
        assert exc_info.value.status_code == 409

        with pytest.raises(HTTPException) as exc_info:
            await visitors.name_cluster(
                request_, cluster_id, NameClusterRequest(person_id="x", name="X")
            )
        assert exc_info.value.status_code == 409

        with pytest.raises(HTTPException) as exc_info:
            await visitors.merge_clusters(request_, cluster_id, cluster_id)
        assert exc_info.value.status_code == 409
