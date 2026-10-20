"""Migration runner: single-runner lock, guarded rollback, status, and the down-file rule.

Lock and rollback tests run against the test database with a temporary migrations directory of
probe migrations (versions 99xx), so the real schema is never rolled back. They skip without
``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import asyncpg
import pytest
import pytest_asyncio

from app.db.migrate import (
    BASELINE_VERSION,
    MigrationError,
    missing_down_migrations,
    rollback,
    run_migrations,
    status,
)
from tests.db import safe_test_dsn

TEST_DSN = safe_test_dsn()
_REPO_MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def _write(directory: Path, version: str, up: str, down: str | None) -> None:
    (directory / f"{version}.up.sql").write_text(up)
    if down is not None:
        (directory / f"{version}.down.sql").write_text(down)


@pytest_asyncio.fixture
async def pool() -> AsyncIterator[asyncpg.Pool]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")
    pool = await asyncpg.create_pool(TEST_DSN, min_size=2, max_size=4)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DROP TABLE IF EXISTS runner_probe_a, runner_probe_b")
        await conn.execute("DELETE FROM alembic_version WHERE version_num LIKE '99%'")
    await pool.close()


def test_every_repository_migration_after_the_baseline_has_a_down_file() -> None:
    assert missing_down_migrations(_REPO_MIGRATIONS) == []


def test_the_scan_reports_a_migration_without_a_down_file(tmp_path: Path) -> None:
    _write(tmp_path, BASELINE_VERSION, "SELECT 1;", None)
    _write(tmp_path, "0002_has_down", "SELECT 1;", "SELECT 1;")
    _write(tmp_path, "0003_no_down", "SELECT 1;", None)
    assert missing_down_migrations(tmp_path) == ["0003_no_down"]


async def test_concurrent_runs_apply_each_file_once(pool: asyncpg.Pool, tmp_path: Path) -> None:
    # The sleep holds the first runner inside the migration; without the lock the second runner
    # would apply the same file and fail on the duplicate table or version row.
    _write(
        tmp_path,
        "9901_probe_a",
        "SELECT pg_sleep(0.3); CREATE TABLE runner_probe_a (id int);",
        "DROP TABLE runner_probe_a;",
    )
    results = await asyncio.gather(run_migrations(pool, tmp_path), run_migrations(pool, tmp_path))
    assert sorted(results) == [0, 1]
    async with pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM alembic_version WHERE version_num = '9901_probe_a'"
            )
            == 1
        )


async def test_rollback_applies_down_files_newest_first(pool: asyncpg.Pool, tmp_path: Path) -> None:
    _write(
        tmp_path,
        "9901_probe_a",
        "CREATE TABLE runner_probe_a (id int);",
        "DROP TABLE runner_probe_a;",
    )
    _write(
        tmp_path,
        "9902_probe_b",
        "CREATE TABLE runner_probe_b (a_id int);",
        "DROP TABLE runner_probe_b;",
    )
    assert await run_migrations(pool, tmp_path) == 2

    assert await rollback(pool, steps=1, migrations_dir=tmp_path) == 1
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT to_regclass('runner_probe_b')") is None
        assert await conn.fetchval("SELECT to_regclass('runner_probe_a')") is not None
    [a, b] = [s for s in await status(pool, tmp_path) if s.version.startswith("99")]
    assert (a.version, a.applied, b.version, b.applied) == (
        "9901_probe_a",
        True,
        "9902_probe_b",
        False,
    )

    # Re-applying after a rollback works: the version row was removed with the table.
    assert await run_migrations(pool, tmp_path) == 1


async def test_rollback_without_a_down_file_refuses_and_changes_nothing(
    pool: asyncpg.Pool, tmp_path: Path
) -> None:
    _write(
        tmp_path,
        "9901_probe_a",
        "CREATE TABLE runner_probe_a (id int);",
        "DROP TABLE runner_probe_a;",
    )
    _write(tmp_path, "9902_probe_b", "CREATE TABLE runner_probe_b (id int);", None)
    await run_migrations(pool, tmp_path)

    with pytest.raises(MigrationError, match=r"no \.down\.sql"):
        await rollback(pool, steps=2, migrations_dir=tmp_path)
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT to_regclass('runner_probe_a')") is not None
        assert await conn.fetchval("SELECT to_regclass('runner_probe_b')") is not None


async def test_rollback_never_reverts_the_baseline(pool: asyncpg.Pool) -> None:
    await run_migrations(pool)
    applied = [s for s in await status(pool) if s.applied]
    with pytest.raises(MigrationError):
        await rollback(pool, steps=len(applied))
    async with pool.acquire() as conn:
        assert await conn.fetchval("SELECT to_regclass('members')") is not None


async def test_invalid_step_counts_are_refused(pool: asyncpg.Pool, tmp_path: Path) -> None:
    with pytest.raises(MigrationError, match="at least 1"):
        await rollback(pool, steps=0, migrations_dir=tmp_path)
    with pytest.raises(MigrationError, match="refusing"):
        await rollback(pool, steps=10_000, migrations_dir=tmp_path)
