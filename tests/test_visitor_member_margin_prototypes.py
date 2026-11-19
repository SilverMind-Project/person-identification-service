"""VisitorStore member-margin uses prototype matching and cluster naming builds prototypes."""

from __future__ import annotations

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.enrollment_repository import PostgresEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.match_policy import MatchThresholds
from app.services.prototype_builder import MemberEmbedding
from app.services.visitor_store import VisitorStore
from tests.db import safe_test_dsn
from tests.visitor_faces import visitor_face

TEST_DSN = safe_test_dsn()


class _FakeMinio:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "image/jpeg") -> str:
        self.uploads[object_name] = data
        return object_name

    def delete_object(self, object_name: str) -> None:
        self.uploads.pop(object_name, None)


class _EmptyEngine:
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return []


def _axis(index: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[index] = 1.0
    return v


@pytest_asyncio.fixture
async def pool():
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    p = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(p)
    yield p
    async with p.acquire() as conn:
        await conn.execute("DELETE FROM visitor_sightings")
        await conn.execute("DELETE FROM visitor_clusters")
        await conn.execute("DELETE FROM face_prototypes")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await p.close()


async def test_member_margin_rejects_face_matching_down_prototype(pool: asyncpg.Pool) -> None:
    repo = PostgresEnrollmentRepository(pool)
    await repo.upsert_member("alice", "Alice")
    # 5 frontal embeddings on axis 0 + 1 downward embedding on axis 1.
    await repo.insert_embeddings(
        "alice",
        [
            *[MemberEmbedding(embedding=_axis(0), pose_bin="frontal") for _ in range(5)],
            MemberEmbedding(embedding=_axis(1), pose_bin="down"),
        ],
    )
    await repo.recompute_centroid("alice")
    await repo.rebuild_prototypes("alice")

    thresholds = MatchThresholds(recognized_min_similarity=0.40, candidate_min_similarity=0.25)
    proto_store = EnrollmentStore(
        repo, _EmptyEngine(), thresholds, gallery_representation="prototypes"
    )
    visitors = VisitorStore(pool, _FakeMinio(), thresholds=thresholds, enrollment_store=proto_store)

    # Probe is close to Alice's down prototype (sim ~ 0.38, inside member_margin 0.05 of 0.40)
    # but near-orthogonal to Alice's frontal centroid (sim < 0.1).
    probe = (0.38 * _axis(1) + np.sqrt(1.0 - 0.38**2) * _axis(9)).astype(np.float32)
    face = visitor_face(probe)
    # Pass a centroid-only IdentifyResult (low similarity 0.08): VisitorStore still queries the
    # active prototype matcher and rejects the face as too close to a member.
    stale_centroid_id = IdentifyResult(
        person_id="unknown",
        name="Guest",
        bbox=face.bbox,
        best_candidate_id="alice",
        similarity=0.08,
        recognition_state="unrecognized",
        gallery_representation="centroid",
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert await visitors.record_sighting(image, face, stale_centroid_id) is None


async def test_naming_visitor_cluster_builds_prototypes_with_null_quality(
    pool: asyncpg.Pool,
) -> None:
    repo = PostgresEnrollmentRepository(pool)
    visitors = VisitorStore(pool, _FakeMinio())
    probe = _axis(7)
    face = visitor_face(probe)
    unknown = IdentifyResult(
        person_id="unknown",
        name="Guest",
        bbox=face.bbox,
        similarity=0.05,
        recognition_state="unrecognized",
    )
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cluster_id = await visitors.record_sighting(image, face, unknown)
    assert cluster_id is not None

    named = await visitors.name_cluster(cluster_id, "nurse-pat", "Nurse Pat")
    assert named.status == "named"

    protos = await repo.list_prototypes("nurse-pat")
    assert len(protos) == 1
    assert protos[0].pose_bin == "frontal"
    assert protos[0].source == "visitor_naming"
    assert protos[0].quality_score is None
    assert protos[0].quality_version is None
