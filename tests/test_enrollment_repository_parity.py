"""InMemoryEnrollmentRepository behaves like PostgresEnrollmentRepository.

Runs the same operations against both and compares every result: members (ordering, names,
counts), existence, embedding counts, centroids, nearest hits (similarity within 1e-5, pgvector
computes in float4), and removal. Skips without ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

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
from app.services.prototype_builder import MemberEmbedding
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
        await conn.execute("DELETE FROM face_prototypes")
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


async def _exercise_prototypes(
    repo: EnrollmentRepository,
) -> tuple[dict[str, object], list[list[float]]]:
    cand_id = str(uuid.UUID(int=42))
    approved_at = datetime(2026, 11, 19, 10, 0, tzinfo=UTC)
    await repo.upsert_member("p-alpha", "Alpha")
    await repo.insert_embeddings(
        "p-alpha",
        [
            MemberEmbedding(
                embedding=_GALLERY["p-alpha"][1][0],
                pose_bin="frontal",
                source="enrollment_photo",
                quality_score=0.8,
                quality_version="heuristic-v1",
            ),
            MemberEmbedding(
                embedding=_GALLERY["p-alpha"][1][1],
                pose_bin=None,  # backfilled to frontal
                source="enrollment_photo",
            ),
        ],
    )
    await repo.upsert_member("p-beta", "Beta")
    await repo.insert_embeddings(
        "p-beta",
        [
            MemberEmbedding(
                embedding=_GALLERY["p-beta"][1][0],
                pose_bin="frontal",
                source="enrollment_photo",
                quality_score=0.7,
                quality_version="heuristic-v1",
            )
        ],
    )
    backfilled = await repo.backfill_missing_pose_bins("frontal")
    await repo.add_member_embedding(
        "p-alpha",
        _GALLERY["p-alpha"][1][2],
        pose_bin="down",
        source="deployment_capture",
        quality_score=0.75,
        quality_version="heuristic-v1",
        source_candidate_id=cand_id,
        approved_by="admin",
        approved_at=approved_at,
        crop_object_key="calibration/42.png",
    )
    await repo.recompute_centroid("p-alpha")
    await repo.recompute_centroid("p-beta")
    await repo.rebuild_prototypes("p-alpha")
    await repo.rebuild_prototypes("p-beta")

    protos_before = await repo.list_prototypes("p-alpha")
    nearest = [await repo.nearest_identities(q, k=2) for q in _QUERIES]
    nearest_ids = [[(h.person_id, h.name, h.gallery_revision) for h in hits] for hits in nearest]
    nearest_sims = [[h.similarity for h in hits] for hits in nearest]

    promoted = sorted(await repo.promoted_candidate_ids())
    await repo.clear_crop_reference("calibration/42.png")
    protos_after_crop_clear = await repo.list_prototypes("p-alpha")

    down_proto = next(p for p in protos_after_crop_clear if p.pose_bin == "down")
    deleted = await repo.delete_prototype("p-alpha", down_proto.prototype_id)
    deleted_missing = await repo.delete_prototype("p-alpha", str(uuid.UUID(int=999)))

    # Attempting to delete beta's sole remaining prototype raises ValueError in both peers.
    beta_proto = (await repo.list_prototypes("p-beta"))[0]
    with pytest.raises(ValueError, match="last embedding"):
        await repo.delete_prototype("p-beta", beta_proto.prototype_id)

    record: dict[str, object] = {
        "backfilled": backfilled,
        "protos_before": [
            (p.pose_bin, p.member_count, p.source, p.crop_object_key) for p in protos_before
        ],
        "protos_after_crop_clear": [
            (p.pose_bin, p.crop_object_key) for p in protos_after_crop_clear
        ],
        "nearest_ids": nearest_ids,
        "promoted": promoted,
        "deleted": [deleted, deleted_missing],
        "revisions": await repo.get_gallery_revisions(),
        "members": [
            (m.person_id, m.gallery_revision, m.prototype_counts) for m in await repo.list_members()
        ],
        "embedding_sources": {
            pid: [(e.pose_bin, e.source, e.source_candidate_id) for e in items]
            for pid, items in (await repo.member_embedding_records()).items()
        },
    }
    return record, nearest_sims


async def test_prototype_methods_match_between_in_memory_and_postgres(
    postgres: PostgresEnrollmentRepository,
) -> None:
    pg_record, pg_sims = await _exercise_prototypes(postgres)
    mem_record, mem_sims = await _exercise_prototypes(InMemoryEnrollmentRepository())

    assert mem_record == pg_record
    assert np.allclose(mem_sims, pg_sims, atol=1e-5)


async def _exercise_version_isolation(
    repo_a: EnrollmentRepository, repo_b: EnrollmentRepository
) -> dict[str, object]:
    await repo_a.upsert_member("p-alpha", "Alpha")
    await repo_a.insert_embeddings(
        "p-alpha",
        [MemberEmbedding(embedding=_GALLERY["p-alpha"][1][0], pose_bin="frontal")],
    )
    await repo_a.recompute_centroid("p-alpha")
    await repo_a.rebuild_prototypes("p-alpha")

    await repo_b.replace_profile_embeddings(
        "p-alpha",
        [MemberEmbedding(embedding=_GALLERY["p-beta"][1][0], pose_bin="left")],
    )

    hits_a_on_a = await repo_a.nearest_centroids(_GALLERY["p-alpha"][1][0], k=2)
    hits_b_on_b = await repo_b.nearest_centroids(_GALLERY["p-beta"][1][0], k=2)
    protos_a = await repo_a.list_prototypes("p-alpha")
    protos_b = await repo_b.list_prototypes("p-alpha")
    return {
        "counts": (
            await repo_a.embedding_count("p-alpha"),
            await repo_b.embedding_count("p-alpha"),
        ),
        "versions": (repo_a.recognizer_version, repo_b.recognizer_version),
        "hits_a": [(h.person_id, round(h.similarity, 4)) for h in hits_a_on_a],
        "hits_b": [(h.person_id, round(h.similarity, 4)) for h in hits_b_on_b],
        "proto_bins_a": [p.pose_bin for p in protos_a],
        "proto_bins_b": [p.pose_bin for p in protos_b],
        "proto_ids_distinct": protos_a[0].prototype_id != protos_b[0].prototype_id,
    }


async def test_recognizer_version_isolation_matches_between_in_memory_and_postgres(
    postgres: PostgresEnrollmentRepository,
) -> None:
    pg_alt = PostgresEnrollmentRepository(
        postgres._pool, recognizer_version="adaface_ir50_webface4m"
    )
    mem_base = InMemoryEnrollmentRepository("buffalo_l")
    mem_alt = mem_base.for_recognizer_version("adaface_ir50_webface4m")

    pg_rec = await _exercise_version_isolation(postgres, pg_alt)
    mem_rec = await _exercise_version_isolation(mem_base, mem_alt)
    assert mem_rec == pg_rec
    assert pg_rec["proto_ids_distinct"] is True
