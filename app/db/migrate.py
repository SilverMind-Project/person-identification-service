"""Database migration runner using asyncpg.

Applies SQL migration files from the ``migrations/`` directory in order.
Tracks applied migrations in an ``alembic_version`` table (Alembic-compatible
convention).  Idempotent — only pending migrations are applied.
"""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


async def run_migrations(pool: asyncpg.Pool) -> int:
    """Apply all pending SQL migrations in order.

    Creates the ``alembic_version`` tracking table if it does not exist,
    scans for ``*.up.sql`` files, and applies any that have not yet been
    recorded.  Each migration runs in its own transaction.

    Returns the number of migrations applied (0 if already up-to-date).
    """
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version (version_num TEXT PRIMARY KEY)"
        )
        rows = await conn.fetch("SELECT version_num FROM alembic_version")
        applied = {r["version_num"] for r in rows}

        if not _MIGRATIONS_DIR.is_dir():
            raise FileNotFoundError(f"Migration directory does not exist: {_MIGRATIONS_DIR}")

        applied_count = 0
        for f in sorted(_MIGRATIONS_DIR.iterdir()):
            if not f.name.endswith(".up.sql"):
                continue
            version = f.name[: -len(".up.sql")]
            if version in applied:
                continue
            sql = f.read_text()
            logger.info("applying_migration version=%s file=%s", version, f.name)
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute("INSERT INTO alembic_version (version_num) VALUES ($1)", version)
            logger.info("migration_applied version=%s", version)
            applied_count += 1

        if applied_count:
            logger.info("migrations_complete count=%d", applied_count)
        return applied_count
