"""InMemoryEnrollmentRepository behaves like PostgresEnrollmentRepository.

Runs the same operations against both and compares every result: members (ordering, names,
counts), existence, embedding counts, centroids, nearest hits (similarity within 1e-5, pgvector
computes in float4), and removal. Skips without ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.enrollment_repository import (
    EnrollmentRepository,
    InMemoryEnrollmentRepository,
    PostgresEnrollmentRepository,
)
from tests.db import safe_test_dsn

TEST_DSN = safe_test_dsn()


@pytest_asyncio.fixture
async def postgres() -> AsyncIterator[PostgresEnrollmentRepository]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield PostgresEnrollmentRepository(pool)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


_RNG = np.random.RandomState(7)
_GALLERY = {
    "p-alpha": ("Alpha", [_unit(_RNG.randn(512)) for _ in range(3)]),
    "p-beta": ("Beta", [_unit(_RNG.randn(512)) for _ in range(2)]),
    "p-gamma": ("Gamma", [_unit(_RNG.randn(512))]),
}
_QUERIES = [_unit(_RNG.randn(512)) for _ in range(4)] + [_GALLERY["p-beta"][1][0]]


async def _exercise(repo: EnrollmentRepository) -> dict[str, object]:
    record: dict[str, object] = {"empty_nearest": await repo.nearest_centroids(_QUERIES[0], k=3)}
    statuses = []
    for person_id, (name, embeddings) in _GALLERY.items():
        statuses.append(await repo.upsert_member(person_id, name))
        await repo.insert_embeddings(person_id, embeddings)
        await repo.recompute_centroid(person_id)
    statuses.append(await repo.upsert_member("p-alpha", "Alpha Renamed"))
    # A member without a centroid never appears among the nearest hits.
    statuses.append(await repo.upsert_member("p-delta", "Delta"))
    record["statuses"] = statuses
    record["members"] = [
        (m.person_id, m.name, m.embedding_count) for m in await repo.list_members()
    ]
    record["exists"] = [await repo.member_exists(pid) for pid in ("p-alpha", "p-zeta")]
    record["counts"] = [await repo.embedding_count(pid) for pid in ("p-alpha", "p-delta", "p-zeta")]
    record["member_count"] = await repo.member_count()
    record["member_embeddings"] = {
        pid: [[round(float(v), 5) for v in e[:8]] for e in embeddings]
        for pid, embeddings in (await repo.member_embeddings()).items()
    }
    beta = await repo.get_member("p-beta")
    record["get"] = (beta.person_id, beta.name, beta.embedding_count) if beta else None
    record["get_missing"] = await repo.get_member("p-zeta")
    record["nearest"] = [
        [(h.person_id, h.name, h.similarity) for h in await repo.nearest_centroids(q, k=2)]
        for q in _QUERIES
    ]
    record["removed"] = [await repo.remove_member("p-gamma"), await repo.remove_member("p-gamma")]
    record["nearest_after_remove"] = [
        h.person_id for h in await repo.nearest_centroids(_GALLERY["p-gamma"][1][0], k=5)
    ]
    return record


def _split(record: dict[str, object]) -> tuple[dict[str, object], list[list[float]]]:
    """Separate similarities (compared with a tolerance) from everything else (compared exactly)."""
    nearest = record.pop("nearest")
    assert isinstance(nearest, list)
    record["nearest_ids"] = [[(pid, name) for pid, name, _ in hits] for hits in nearest]
    return record, [[sim for _, _, sim in hits] for hits in nearest]


async def test_in_memory_repository_matches_postgres(
    postgres: PostgresEnrollmentRepository,
) -> None:
    pg_record, pg_similarities = _split(await _exercise(postgres))
    mem_record, mem_similarities = _split(await _exercise(InMemoryEnrollmentRepository()))

    assert mem_record == pg_record
    assert np.allclose(mem_similarities, pg_similarities, atol=1e-5)


async def test_centroid_is_the_normalized_mean_in_both(
    postgres: PostgresEnrollmentRepository,
) -> None:
    name, embeddings = _GALLERY["p-alpha"]
    expected = np.mean(embeddings, axis=0)
    expected /= np.linalg.norm(expected)
    for repo in (postgres, InMemoryEnrollmentRepository()):
        await repo.upsert_member("p-alpha", name)
        await repo.insert_embeddings("p-alpha", embeddings)
        assert np.allclose(await repo.recompute_centroid("p-alpha"), expected, atol=1e-6)
