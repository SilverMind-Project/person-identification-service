"""Database migration runner using asyncpg.

Applies SQL migration files from the ``migrations/`` directory in order. Tracks applied migrations
in an ``alembic_version`` table (Alembic-compatible convention). Idempotent: only pending
migrations are applied.

Every run, rollback, and status read holds a Postgres advisory lock, so two replicas starting
together never apply the same file twice. Migrations are incremental: every migration after the
frozen baseline ships a ``.down.sql`` that exactly reverses it, and ``rollback`` refuses to touch
anything without one.

Operator CLI (reads ``database.dsn`` from settings, so ``DATABASE_URL`` in the environment):

    python -m app.db.migrate status
    python -m app.db.migrate rollback --steps 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from app import config

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"
# The frozen baseline has no down migration: rolling it back would drop household data.
BASELINE_VERSION = "0001_initial_schema"
_LOCK_KEY = "person_id_migrations"
_UP = ".up.sql"
_DOWN = ".down.sql"


class MigrationError(RuntimeError):
    """A migration operation was refused; nothing was changed."""


@dataclass(frozen=True)
class MigrationStatus:
    version: str
    applied: bool
    has_file: bool
    has_down: bool


def _versions(migrations_dir: Path) -> list[str]:
    if not migrations_dir.is_dir():
        raise FileNotFoundError(f"Migration directory does not exist: {migrations_dir}")
    return sorted(f.name[: -len(_UP)] for f in migrations_dir.iterdir() if f.name.endswith(_UP))


def missing_down_migrations(migrations_dir: Path = _MIGRATIONS_DIR) -> list[str]:
    """Versions after the frozen baseline that have no ``.down.sql``."""
    return [
        version
        for version in _versions(migrations_dir)
        if version != BASELINE_VERSION and not (migrations_dir / f"{version}{_DOWN}").is_file()
    ]


@asynccontextmanager
async def _locked(pool: asyncpg.Pool) -> AsyncIterator[asyncpg.Connection]:
    """A connection holding the session-level migration lock for the duration."""
    async with pool.acquire() as conn:
        await conn.execute("SELECT pg_advisory_lock(hashtext($1))", _LOCK_KEY)
        try:
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS alembic_version (version_num TEXT PRIMARY KEY)"
            )
            yield conn
        finally:
            await conn.execute("SELECT pg_advisory_unlock(hashtext($1))", _LOCK_KEY)


async def _applied(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT version_num FROM alembic_version")
    return {r["version_num"] for r in rows}


async def run_migrations(pool: asyncpg.Pool, migrations_dir: Path = _MIGRATIONS_DIR) -> int:
    """Apply all pending SQL migrations in order, each in its own transaction.

    Returns the number of migrations applied (0 if already up to date).
    """
    versions = _versions(migrations_dir)
    async with _locked(pool) as conn:
        applied = await _applied(conn)
        applied_count = 0
        for version in versions:
            if version in applied:
                continue
            sql = (migrations_dir / f"{version}{_UP}").read_text()
            logger.info("applying_migration version=%s", version)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO alembic_version (version_num) VALUES ($1)", version)
            logger.info("migration_applied version=%s", version)
            applied_count += 1

    if applied_count:
        logger.info("migrations_complete count=%d", applied_count)
    return applied_count


async def rollback(
    pool: asyncpg.Pool, steps: int = 1, migrations_dir: Path = _MIGRATIONS_DIR
) -> int:
    """Revert the ``steps`` most recently applied migrations, newest first.

    Every target is checked before anything runs: the baseline, a version whose files are gone,
    or a version without a ``.down.sql`` refuses the whole rollback. Each down migration runs in
    its own transaction together with removing its version row. Returns the number reverted.
    """
    if steps < 1:
        raise MigrationError("rollback steps must be at least 1")
    async with _locked(pool) as conn:
        applied = sorted(await _applied(conn), reverse=True)
        if steps > len(applied):
            raise MigrationError(f"only {len(applied)} migrations are applied; refusing {steps}")
        targets = applied[:steps]
        for version in targets:
            if version == BASELINE_VERSION:
                raise MigrationError(f"{version} is the frozen baseline and cannot be rolled back")
            if not (migrations_dir / f"{version}{_DOWN}").is_file():
                raise MigrationError(f"{version} has no {_DOWN} file; refusing to roll back")
        for version in targets:
            sql = (migrations_dir / f"{version}{_DOWN}").read_text()
            logger.info("reverting_migration version=%s", version)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("DELETE FROM alembic_version WHERE version_num = $1", version)
            logger.info("migration_reverted version=%s", version)
    return len(targets)


async def status(
    pool: asyncpg.Pool, migrations_dir: Path = _MIGRATIONS_DIR
) -> list[MigrationStatus]:
    """Every known version, from files and from the tracking table, in order."""
    versions = set(_versions(migrations_dir))
    async with _locked(pool) as conn:
        applied = await _applied(conn)
    return [
        MigrationStatus(
            version=version,
            applied=version in applied,
            has_file=version in versions,
            has_down=(migrations_dir / f"{version}{_DOWN}").is_file(),
        )
        for version in sorted(versions | applied)
    ]


async def _cli(args: argparse.Namespace) -> int:
    dsn = config.get("database.dsn")
    if not dsn:
        raise SystemExit("database.dsn is not configured (set DATABASE_URL)")
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1)
    try:
        if args.command == "status":
            for s in await status(pool):
                state = "applied" if s.applied else "pending"
                notes = [] if s.has_file else ["no file (squashed or removed)"]
                if s.has_file and s.version != BASELINE_VERSION and not s.has_down:
                    notes.append("no down file")
                sys.stdout.write(f"{s.version:<40} {state:<8} {', '.join(notes)}\n")
            return 0
        reverted = await rollback(pool, steps=args.steps)
        sys.stdout.write(f"reverted {reverted} migration(s)\n")
        return 0
    finally:
        await pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="person-identification-service migrations")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="list migrations and whether each is applied")
    rollback_parser = sub.add_parser("rollback", help="revert the most recent migrations")
    rollback_parser.add_argument("--steps", type=int, default=1)
    try:
        return asyncio.run(_cli(parser.parse_args(argv)))
    except MigrationError as exc:
        sys.stderr.write(f"refused: {exc}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
