"""Golden characterization of enrollment and identification against a real database.

Freezes what ``EnrollmentStore`` returns for a seeded synthetic gallery: enroll results, member
listings, and every ``IdentifyResult`` field (including the geometry and quality carry-forward)
for query embeddings placed at chosen cosine similarities around both recognition thresholds.
A refactor of the store, repository, or matching policy must leave the golden file unchanged.

Regenerate only for an intended behavior change:
``PID_REGENERATE_GOLDEN=1 TEST_DATABASE_URL=... pytest tests/test_identify_characterization.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.enrollment_repository import PostgresEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality
from tests.db import safe_test_dsn
from tests.gallery import DEFAULT_THRESHOLDS

TEST_DSN = safe_test_dsn()
_GOLDEN = Path(__file__).parent / "fixtures" / "identify_characterization_golden.json"
_DIM = 512
_QUALITY = FaceQuality(
    score=0.61, resolution=0.7, sharpness=0.8, detection=0.9, pitch=1.0, version="heuristic-v1"
)
# Five SCRFD keypoints, eyes 30 px apart: geometry is derived from these and the box.
_KEYPOINTS = np.array(
    [[40.0, 50.0], [70.0, 50.0], [55.0, 65.0], [45.0, 80.0], [65.0, 80.0]], dtype=np.float32
)


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _member_embeddings(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    base = rng.randn(_DIM)
    return [_unit(base + 0.15 * rng.randn(_DIM)) for _ in range(count)]


_MEMBERS = {
    "resident-alpha": ("Resident Alpha", _member_embeddings(1, 3)),
    "resident-beta": ("Resident Beta", _member_embeddings(2, 2)),
    "resident-gamma": ("Resident Gamma", _member_embeddings(3, 1)),
}


def _at_similarity(target: np.ndarray, similarity: float, seed: int) -> np.ndarray:
    """A unit vector whose cosine similarity to unit ``target`` is ``similarity``."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(_DIM)
    orthogonal = _unit(noise - float(np.dot(noise, target)) * target)
    return _unit(similarity * target + math.sqrt(1.0 - similarity**2) * orthogonal)


class _ScriptedEngine:
    """Returns the next scripted embedding as one detected face per image."""

    def __init__(self) -> None:
        self.queue: list[np.ndarray] = []

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        if not self.queue:
            return []
        return [_face(self.queue.pop(0))]


def _face(embedding: np.ndarray) -> DetectedFace:
    face = DetectedFace(
        bbox=[20.0, 30.0, 90.0, 110.0],
        embedding=embedding,
        det_score=0.93,
        keypoints_5_px=_KEYPOINTS,
        yaw_deg=12.5,
        pitch_deg=-7.0,
        roll_deg=1.5,
    )
    face.quality = _QUALITY
    return face


@pytest_asyncio.fixture
async def pool():
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


def _store(pool: asyncpg.Pool, engine: _ScriptedEngine) -> EnrollmentStore:
    return EnrollmentStore(PostgresEnrollmentRepository(pool), engine, DEFAULT_THRESHOLDS)


def _result_record(result: Any) -> dict[str, Any]:
    record = dataclasses.asdict(result)
    record["similarity"] = round(record["similarity"], 5)
    return record


async def _record(pool: asyncpg.Pool) -> dict[str, Any]:
    engine = _ScriptedEngine()
    store = _store(pool, engine)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    record: dict[str, Any] = {}

    record["empty_gallery"] = [
        _result_record(r) for r in await store.identify_all([_face(_unit(np.ones(_DIM)))])
    ]
    record["member_count_empty"] = await store.member_count()

    enrolls = []
    for person_id, (name, embeddings) in _MEMBERS.items():
        engine.queue = list(embeddings)
        result = await store.enroll(person_id, name, [image] * len(embeddings))
        enrolls.append(result.model_dump())
    # Re-enroll alpha with one good image and one image without a face.
    engine.queue = [_MEMBERS["resident-alpha"][1][0]]
    enrolls.append(
        (await store.enroll("resident-alpha", "Alpha Renamed", [image, image])).model_dump()
    )
    # A brand-new member whose only image has no face.
    engine.queue = []
    enrolls.append((await store.enroll("resident-delta", "Delta", [image])).model_dump())
    record["enroll"] = enrolls

    members = await store.list_members()
    record["members"] = [
        {"person_id": m.person_id, "name": m.name, "embedding_count": m.embedding_count}
        for m in members
    ]
    beta = await store.get_member("resident-beta")
    record["get_beta"] = None if beta is None else [beta.person_id, beta.name, beta.embedding_count]
    record["get_missing"] = await store.get_member("nobody") is None
    record["member_count"] = await store.member_count()

    async with pool.acquire() as conn:
        centroids = {
            r["person_id"]: np.asarray(r["centroid"], dtype=np.float32)
            for r in await conn.fetch("SELECT person_id, centroid FROM centroids")
        }
    queries = []
    for index, (person_id, similarity) in enumerate(
        [
            ("resident-alpha", 0.95),
            ("resident-alpha", 0.41),
            ("resident-beta", 0.39),
            ("resident-beta", 0.26),
            ("resident-gamma", 0.24),
            ("resident-gamma", 0.10),
            ("resident-gamma", -0.30),
        ]
    ):
        queries.append(_face(_at_similarity(centroids[person_id], similarity, seed=100 + index)))
    # Opposite every centroid: the nearest similarity is negative and is reported unclamped.
    queries.append(_face(_unit(-sum(centroids.values()))))
    record["identify"] = [_result_record(r) for r in await store.identify_all(queries)]

    record["remove_beta"] = await store.remove_member("resident-beta")
    record["remove_beta_again"] = await store.remove_member("resident-beta")
    record["member_count_after_remove"] = await store.member_count()
    return record


async def test_enrollment_and_identification_match_golden(pool: asyncpg.Pool) -> None:
    actual = json.dumps(await _record(pool), indent=1, sort_keys=True)
    if os.environ.get("PID_REGENERATE_GOLDEN") == "1":
        _GOLDEN.write_text(actual + "\n")
    assert actual == _GOLDEN.read_text().rstrip("\n"), (
        "identify or enroll output changed; a refactor must not change it"
    )
