"""Calibration candidate repository: Postgres and in-memory peers behave the same.

Covers insert and round trip, keyset pagination, label transitions (including the database's
label-completeness check), purge keeping labeled rows, bucket counts, and deletion. Skips without
``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.services.calibration_candidates import (
    CalibrationCandidate,
    CalibrationCandidateRepository,
    InMemoryCalibrationCandidateRepository,
    PostgresCalibrationCandidateRepository,
)
from tests.db import safe_test_dsn

TEST_DSN = safe_test_dsn()
_T = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


@pytest_asyncio.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM calibration_candidates")
    await pool.close()


def _candidate(index: int, **changes: Any) -> CalibrationCandidate:
    rng = np.random.RandomState(index)
    embedding = rng.randn(512).astype(np.float32)
    fields: dict[str, Any] = {
        "candidate_id": str(uuid.UUID(int=index + 1)),
        "captured_at": _T + timedelta(minutes=index),
        "camera_id": f"cam-{index % 2}",
        "source": "cts_identify",
        "crop_object_key": f"calibration/2026-10-21/{index}.png",
        "embedding": embedding / np.linalg.norm(embedding),
        "arcface_model_version": "buffalo_l",
        "preprocessing_version": "v1",
        "quality_version": "heuristic-v1",
        "quality_score": 0.05 + 0.1 * (index % 10),
        "quality_components": {
            "resolution": 0.5,
            "sharpness": 0.25,
            "detection": 1.0,
            "pitch": 0.75,
        },
        "capture_profile": {
            "crop_pad_top_frac": 0.15,
            "crop_pad_side_frac": 0.1,
            "crop_jpeg_quality": 95,
        },
        "interocular_px": None if index == 3 else 20.0 + index,
        "yaw_deg": -5.0,
        "pitch_deg": 12.5,
        "nearest_person_id": None if index == 4 else "resident-alpha",
        "nearest_similarity": -0.125 if index == 4 else 0.375,
    }
    fields.update(changes)
    return CalibrationCandidate(**fields)


def _plain(c: CalibrationCandidate) -> dict[str, Any]:
    record = {k: getattr(c, k) for k in c.__dataclass_fields__}
    record["embedding"] = [round(float(v), 5) for v in c.embedding]
    record["quality_components"] = dict(c.quality_components)
    record["capture_profile"] = dict(c.capture_profile)
    return record


async def _exercise(repo: CalibrationCandidateRepository) -> dict[str, Any]:
    for index in range(7):
        await repo.insert(_candidate(index))
    out: dict[str, Any] = {}
    first = await repo.list_page(state=None, limit=3, cursor=None)
    second = await repo.list_page(state=None, limit=3, cursor=first.next_cursor)
    third = await repo.list_page(state=None, limit=3, cursor=second.next_cursor)
    out["pages"] = [[c.candidate_id for c in page.candidates] for page in (first, second, third)]
    out["last_cursor"] = third.next_cursor
    out["round_trip"] = _plain(await repo.get(_candidate(3).candidate_id))  # type: ignore[arg-type]
    out["missing"] = await repo.get(str(uuid.UUID(int=999)))

    labeled_at = _T + timedelta(days=1)
    member = await repo.label(
        _candidate(0).candidate_id,
        kind="member",
        identity_id="resident-alpha",
        source="operator_reviewed",
        labeled_by="op",
        labeled_at=labeled_at,
    )
    visitor = await repo.label(
        _candidate(1).candidate_id,
        kind="not_enrolled",
        identity_id=None,
        source="operator_reviewed",
        labeled_by="op",
        labeled_at=labeled_at,
    )
    # Relabel then reject: a reject clears every label field.
    await repo.label(
        _candidate(2).candidate_id,
        kind="not_a_face",
        identity_id=None,
        source="operator_reviewed",
        labeled_by="op",
        labeled_at=labeled_at,
    )
    rejected = await repo.reject(_candidate(2).candidate_id)
    out["labels"] = [_plain(c) if c else None for c in (member, visitor, rejected)]
    out["label_missing"] = await repo.label(
        str(uuid.UUID(int=999)),
        kind="not_a_face",
        identity_id=None,
        source="operator_reviewed",
        labeled_by="op",
        labeled_at=labeled_at,
    )
    out["unlabeled_page"] = [
        c.candidate_id
        for c in (await repo.list_page(state="unlabeled", limit=10, cursor=None)).candidates
    ]
    out["by_state"] = await repo.count_by_state()
    out["buckets"] = sorted((await repo.bucket_counts(_T + timedelta(minutes=3))).items())
    out["labeled"] = [c.candidate_id for c in await repo.list_labeled()]
    out["purged"] = await repo.purge(_T + timedelta(minutes=4))
    out["after_purge"] = await repo.count_by_state()
    out["deleted"] = [
        await repo.delete(_candidate(0).candidate_id),
        await repo.delete(_candidate(0).candidate_id),
    ]
    return out


async def test_in_memory_repository_matches_postgres(pool: asyncpg.Pool) -> None:
    postgres = await _exercise(PostgresCalibrationCandidateRepository(pool))
    in_memory = await _exercise(InMemoryCalibrationCandidateRepository())
    assert in_memory == postgres
    # Purge removes the old unlabeled and rejected rows (indexes 2 and 3) and keeps labeled ones.
    assert postgres["purged"] == ["calibration/2026-10-21/2.png", "calibration/2026-10-21/3.png"]
    assert postgres["after_purge"] == {"unlabeled": 3, "labeled": 2, "rejected": 0}


async def test_database_rejects_an_incomplete_label(pool: asyncpg.Pool) -> None:
    repo = PostgresCalibrationCandidateRepository(pool)
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.insert(_candidate(0, label_identity_id="resident-alpha"))
    with pytest.raises(asyncpg.CheckViolationError):
        await repo.insert(_candidate(1, label_state="labeled", label_kind="member"))
