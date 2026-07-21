"""Tests for VisitorStore: ingest gates, clustering, review, retention.

Identity-continuity M06.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.enrollment_store import EnrollmentStore
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
        if object_name not in self.uploads:
            raise KeyError(object_name)
        del self.uploads[object_name]


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _embedding_at_similarity(ref: np.ndarray, target_sim: float, seed: int = 7) -> np.ndarray:
    """Return an embedding at approximately *target_sim* cosine similarity to *ref*."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(512).astype(np.float32)
    parallel = np.dot(noise, ref) * ref
    perp = noise - parallel
    if np.linalg.norm(perp) < 1e-8:
        perp = rng.randn(512).astype(np.float32)
    perp = perp / np.linalg.norm(perp)
    sim = max(-1.0, min(1.0, target_sim))
    ortho_weight = np.sqrt(max(0.0, 1.0 - sim * sim))
    result = sim * ref + ortho_weight * perp
    return (result / np.linalg.norm(result)).astype(np.float32)


def _face(embedding: np.ndarray, det_score: float = 0.95, bbox=None) -> DetectedFace:
    return DetectedFace(
        bbox=bbox or [10.0, 20.0, 110.0, 140.0],
        embedding=embedding,
        det_score=det_score,
    )


def _unknown_identity(similarity: float = 0.1, best_candidate_id: str | None = None) -> IdentifyResult:
    return IdentifyResult(
        person_id="unknown",
        name="Guest",
        confidence=max(0.0, similarity),
        bbox=[10.0, 20.0, 110.0, 140.0],
        best_candidate_id=best_candidate_id,
        similarity=similarity,
        recognition_state="unrecognized" if similarity < 0.25 else "candidate",
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
async def minio():
    return _FakeMinioClient()


@pytest_asyncio.fixture
async def store(pool, minio):
    return VisitorStore(pool, minio)


async def _cluster_rows(pool):
    async with pool.acquire() as conn:
        return await conn.fetch("SELECT * FROM visitor_clusters")


async def _sighting_rows(pool, cluster_id):
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT * FROM visitor_sightings WHERE cluster_id = $1 ORDER BY seen_at", cluster_id
        )


class TestIngestGates:
    async def test_unknown_face_below_gates_not_stored(self, store, pool):
        """Each gate rejects independently and stores nothing."""
        emb = _embedding(1)

        # Low detection score
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8),
            _face(emb, det_score=0.3),
            _unknown_identity(similarity=0.05),
        )
        assert cluster_id is None

        # Face too small
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8),
            _face(emb, det_score=0.9, bbox=[10.0, 10.0, 30.0, 30.0]),
            _unknown_identity(similarity=0.05),
        )
        assert cluster_id is None

        assert len(await _cluster_rows(pool)) == 0

    async def test_near_member_face_not_stored(self, store, pool):
        """member-margin gate: similarity within margin of the recognition threshold fails closed."""
        emb = _embedding(2)
        # recognition.threshold default 0.4, member_margin default 0.05 -> 0.35 cutoff
        identity = _unknown_identity(similarity=0.36, best_candidate_id="alice")

        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(emb), identity
        )
        assert cluster_id is None
        assert len(await _cluster_rows(pool)) == 0

    async def test_far_from_member_face_is_stored(self, store, pool):
        """A face safely below the member margin passes the gate."""
        emb = _embedding(3)
        identity = _unknown_identity(similarity=0.1, best_candidate_id="alice")

        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(emb), identity
        )
        assert cluster_id is not None
        assert len(await _cluster_rows(pool)) == 1


class TestClusterAssignment:
    async def test_new_cluster_created_then_joined(self, store, pool):
        """Two similar embeddings join one cluster; an orthogonal one creates a second."""
        base = _embedding(10)
        similar = _embedding_at_similarity(base, target_sim=0.8, seed=11)
        orthogonal = _embedding_at_similarity(base, target_sim=0.0, seed=12)

        cluster_1 = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(base), _unknown_identity()
        )
        cluster_2 = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(similar), _unknown_identity()
        )
        cluster_3 = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(orthogonal), _unknown_identity()
        )

        assert cluster_1 == cluster_2
        assert cluster_3 != cluster_1

        rows = await _cluster_rows(pool)
        assert len(rows) == 2


class TestCentroidAndSurfacing:
    async def test_centroid_running_mean_normalized(self, store, pool):
        base = _embedding(20)
        similar = _embedding_at_similarity(base, target_sim=0.9, seed=21)

        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(base), _unknown_identity()
        )
        await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(similar), _unknown_identity()
        )

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT centroid, sighting_count FROM visitor_clusters WHERE cluster_id = $1",
                cluster_id,
            )
        centroid = np.asarray(row["centroid"], dtype=np.float32)
        assert row["sighting_count"] == 2
        assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-4

        expected = (base + similar) / 2
        expected = expected / np.linalg.norm(expected)
        assert np.allclose(centroid, expected, atol=1e-4)

    async def test_distinct_days_and_surfacing(self, store, pool):
        """3 sightings on 3 distinct days surfaces; 3 same-day sightings do not."""
        base = _embedding(30)
        now = datetime.now(UTC)

        cluster_id = None
        for days_ago in (2, 1, 0):
            emb = _embedding_at_similarity(base, target_sim=0.9, seed=30 + days_ago)
            cluster_id = await store.record_sighting(
                np.zeros((200, 200, 3), dtype=np.uint8),
                _face(emb),
                _unknown_identity(),
                seen_at=now - timedelta(days=days_ago),
            )

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status, distinct_days FROM visitor_clusters WHERE cluster_id = $1",
                cluster_id,
            )
        assert row["distinct_days"] == 3
        assert row["status"] == "surfaced"

        # A separate cluster seeded 3 times in a single day never surfaces.
        base2 = _embedding(40)
        cluster_2 = None
        for i in range(3):
            emb = _embedding_at_similarity(base2, target_sim=0.9, seed=40 + i)
            cluster_2 = await store.record_sighting(
                np.zeros((200, 200, 3), dtype=np.uint8),
                _face(emb),
                _unknown_identity(),
                seen_at=now,
            )
        async with pool.acquire() as conn:
            row2 = await conn.fetchrow(
                "SELECT status, distinct_days FROM visitor_clusters WHERE cluster_id = $1",
                cluster_2,
            )
        assert row2["distinct_days"] == 1
        assert row2["status"] == "candidate"


class TestDismissAndRetention:
    async def test_dismissed_absorbs_but_never_resurfaces(self, store, pool):
        base = _embedding(50)
        now = datetime.now(UTC)

        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(base), _unknown_identity(), seen_at=now
        )
        assert await store.dismiss_cluster(cluster_id) is True

        # Further matching sightings still absorb into the dismissed cluster...
        for days_ago in (2, 1):
            emb = _embedding_at_similarity(base, target_sim=0.9, seed=51 + days_ago)
            joined = await store.record_sighting(
                np.zeros((200, 200, 3), dtype=np.uint8),
                _face(emb),
                _unknown_identity(),
                seen_at=now - timedelta(days=days_ago),
            )
            assert joined == cluster_id

        # ...but the cluster never re-promotes to surfaced despite 3 distinct days.
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status, distinct_days FROM visitor_clusters WHERE cluster_id = $1",
                cluster_id,
            )
        assert row["distinct_days"] == 3
        assert row["status"] == "dismissed"

    async def test_retention_purges_unnamed_and_crops_spares_named(self, store, pool, minio):
        old = datetime.now(UTC) - timedelta(days=61)
        recent = datetime.now(UTC)

        old_cluster = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8),
            _face(_embedding(60)),
            _unknown_identity(),
            seen_at=old,
        )
        recent_cluster = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8),
            _face(_embedding(61)),
            _unknown_identity(),
            seen_at=recent,
        )
        named_cluster = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8),
            _face(_embedding(62)),
            _unknown_identity(),
            seen_at=old,
        )
        await store.name_cluster(named_cluster, "nurse-priya", "Nurse Priya")

        assert minio.uploads, "expected crops to have been uploaded"

        purged = await store.purge_expired()
        assert purged == 1

        remaining_ids = {str(r["cluster_id"]) for r in await _cluster_rows(pool)}
        assert old_cluster not in remaining_ids
        assert recent_cluster in remaining_ids
        assert named_cluster in remaining_ids

        old_crop_keys = [f"visitor-crops/{old_cluster}/" in k for k in minio.uploads]
        assert not any(old_crop_keys), "expired cluster's crop should have been deleted"


class TestNamingAndMerge:
    async def test_name_cluster_creates_member_and_recognizable(self, store, pool):
        emb = _embedding(70)
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(emb), _unknown_identity()
        )

        result = await store.name_cluster(cluster_id, "nurse-priya", "Nurse Priya")
        assert result.named_person_id == "nurse-priya"
        assert result.embedding_count == 1

        async with pool.acquire() as conn:
            member = await conn.fetchrow(
                "SELECT * FROM members WHERE person_id = $1", "nurse-priya"
            )
            centroid_row = await conn.fetchrow(
                "SELECT centroid FROM centroids WHERE person_id = $1", "nurse-priya"
            )
        assert member is not None
        assert centroid_row is not None
        centroid = np.asarray(centroid_row["centroid"], dtype=np.float32)
        assert np.allclose(centroid, emb, atol=1e-4)

    async def test_name_cluster_rejects_existing_person_id(self, store, pool):
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO members (person_id, name) VALUES ($1, $2)", "alice", "Alice"
            )
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(80)), _unknown_identity()
        )
        with pytest.raises(ValueError, match="already exists"):
            await store.name_cluster(cluster_id, "alice", "Alice Again")

    async def test_merge_clusters_recomputes_centroid(self, store, pool):
        base = _embedding(90)
        orthogonal = _embedding_at_similarity(base, target_sim=0.0, seed=91)

        cluster_a = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(base), _unknown_identity()
        )
        cluster_b = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(orthogonal), _unknown_identity()
        )
        assert cluster_a != cluster_b

        merged = await store.merge_clusters(cluster_a, cluster_b)
        remaining = await _cluster_rows(pool)
        assert len(remaining) == 1
        assert str(remaining[0]["cluster_id"]) == merged.cluster_id
        assert merged.sighting_count == 2

        expected = (base + orthogonal) / 2
        expected = expected / np.linalg.norm(expected)
        centroid = np.asarray(remaining[0]["centroid"], dtype=np.float32)
        assert np.allclose(centroid, expected, atol=1e-4)

    async def test_merge_cluster_with_itself_rejected(self, store):
        cluster_id = await store.record_sighting(
            np.zeros((200, 200, 3), dtype=np.uint8), _face(_embedding(95)), _unknown_identity()
        )
        with pytest.raises(ValueError, match="itself"):
            await store.merge_clusters(cluster_id, cluster_id)


class TestMilestoneCompletionCriterion:
    async def test_synthetic_nurse_scenario_closes_loop(self, store, pool):
        """M06 completion criterion, verbatim: one recurring nurse (seen on 3
        distinct days) plus two one-off couriers yields exactly one surfaced
        cluster; naming it creates a member that identify() subsequently
        recognizes as that person_id."""
        now = datetime.now(UTC)
        nurse_base = _embedding(100)

        for days_ago in (2, 1, 0):
            emb = _embedding_at_similarity(nurse_base, target_sim=0.9, seed=100 + days_ago)
            await store.record_sighting(
                np.zeros((200, 200, 3), dtype=np.uint8),
                _face(emb),
                _unknown_identity(),
                seen_at=now - timedelta(days=days_ago),
            )

        # Two unrelated one-off couriers: distinct clusters, never surfaced.
        for seed in (200, 300):
            await store.record_sighting(
                np.zeros((200, 200, 3), dtype=np.uint8),
                _face(_embedding(seed)),
                _unknown_identity(),
                seen_at=now,
            )

        surfaced = await store.list_clusters(status="surfaced")
        assert len(surfaced) == 1
        assert surfaced[0].distinct_days == 3

        candidates = await store.list_clusters(status="candidate")
        assert len(candidates) == 2

        result = await store.name_cluster(surfaced[0].cluster_id, "nurse-priya", "Nurse Priya")
        assert result.status == "named"

        class _NoOpFaceEngine:
            async def detect_faces(self, image):
                return []

        enrollment_store = EnrollmentStore(pool, _NoOpFaceEngine())
        identified = await enrollment_store.identify(nurse_base)
        assert identified.person_id == "nurse-priya"
        assert identified.recognition_state == "recognized"
