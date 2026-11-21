"""Migration 0006 (attested label state): it applies, rolls back, and refuses to lose attested rows.

Runs against the test database only (``TEST_DATABASE_URL``, a ``*_test`` database).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import rollback, run_migrations, status
from tests.db import safe_test_dsn

TEST_DSN = safe_test_dsn()

_INSERT_ATTESTED = """INSERT INTO calibration_candidates (
    candidate_id, captured_at, camera_id, source, crop_object_key, embedding,
    arcface_model_version, preprocessing_version, quality_version, quality_score,
    quality_components, capture_profile, yaw_deg, pitch_deg, nearest_similarity, label_state,
    label_kind, label_identity_id, label_source, labeled_by, labeled_at,
    attestation_association_id, attestation_grade)
VALUES ($1::uuid, now(), 'cam-1', 'cts_identify', 'k', $2, 'buffalo_l', 'v1', 'heuristic-v1',
        0.5, '{}'::jsonb, '{}'::jsonb, 0, 0, 0.4, 'attested', 'member', 'resident-alpha',
        'sensor_attested', 'wearable:seg-1', now(), 'seg-1', 'room')"""


@pytest_asyncio.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=2)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM calibration_candidates")
    await run_migrations(pool)
    await pool.close()


async def _head(pool: asyncpg.Pool) -> str:
    return [s.version for s in await status(pool) if s.applied][-1]


async def _steps_to_0006(pool: asyncpg.Pool) -> int:
    applied = [s.version for s in await status(pool) if s.applied]
    return len(applied) - applied.index("0006_attested_label_state")


async def test_down_refuses_while_attested_rows_exist(pool: asyncpg.Pool) -> None:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    async with pool.acquire() as conn:
        await conn.execute(_INSERT_ATTESTED, str(uuid.uuid4()), embedding)
    head = await _head(pool)
    with pytest.raises(asyncpg.RaiseError, match="attested rows"):
        await rollback(pool, steps=await _steps_to_0006(pool))
    assert await _head(pool) == head


async def test_down_and_up_round_trip_without_attested_rows(pool: asyncpg.Pool) -> None:
    await rollback(pool, steps=await _steps_to_0006(pool))
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'calibration_candidates' AND column_name = 'audit_verdict'"
            )
            == 0
        )
    await run_migrations(pool)
    assert "0006_attested_label_state" in [s.version for s in await status(pool) if s.applied]
