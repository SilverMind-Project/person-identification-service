"""Backfill missing pose bins and rebuild pose-binned prototypes for enrolled members.

Usage:
    DATABASE_URL=postgresql://... uv run python -m app.tools.rebuild_prototypes
    DATABASE_URL=postgresql://... uv run python -m app.tools.rebuild_prototypes --person-id alice

Idempotent: embeddings whose ``pose_bin`` is NULL receive ``frontal`` (their source images are not
retained in object storage), existing pose bins are preserved, and ``face_prototypes`` is rebuilt
through ``PostgresEnrollmentRepository`` using the configured ``prototypes`` policy.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg
from pgvector.asyncpg import register_vector

from app import config
from app.services.enrollment_repository import EnrollmentRepository, PostgresEnrollmentRepository
from app.services.prototype_builder import PrototypePolicy


async def rebuild_all_prototypes(
    repository: EnrollmentRepository,
    *,
    policy: PrototypePolicy = PrototypePolicy(),
    person_id: str | None = None,
) -> tuple[int, dict[str, int]]:
    """Backfill NULL pose bins to ``frontal`` and rebuild prototypes for one or every member.

    Returns ``(backfilled_embeddings, {person_id: prototype_count})``.
    """
    backfilled = await repository.backfill_missing_pose_bins("frontal")
    if person_id is not None:
        if not await repository.member_exists(person_id):
            raise LookupError(f"member '{person_id}' not found")
        target_ids = [person_id]
    else:
        target_ids = [m.person_id for m in await repository.list_members()]

    counts: dict[str, int] = {}
    for pid in target_ids:
        built = await repository.rebuild_prototypes(pid, policy)
        counts[pid] = len(built)
    return backfilled, counts


async def _run(args: argparse.Namespace) -> int:
    dsn = config.get("database.dsn")
    if not dsn:
        print("FAIL: database.dsn is not configured (set DATABASE_URL)")
        return 1

    proto_cfg = config.get("prototypes", {}) or {}
    policy = PrototypePolicy(
        min_members_per_bin=int(proto_cfg.get("min_members_per_bin", 3)),
        max_per_bin=int(proto_cfg.get("max_per_bin", 6)),
        missing_quality_weight=float(proto_cfg.get("missing_quality_weight", 0.5)),
        expected_quality_version=str(config.get("face_quality.scorer", "heuristic-v1")),
        min_promotion_quality=float(proto_cfg.get("min_promotion_quality", 0.5)),
        min_promotion_interocular_px=float(proto_cfg.get("min_promotion_interocular_px", 24.0)),
    )

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(dsn, init=_init, min_size=1, max_size=2)
    try:
        repo = PostgresEnrollmentRepository(pool)
        backfilled, counts = await rebuild_all_prototypes(
            repo, policy=policy, person_id=args.person_id
        )
    except LookupError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        await pool.close()

    print(f"backfilled_pose_bins={backfilled} members_rebuilt={len(counts)}")
    for pid, count in sorted(counts.items()):
        print(f"  {pid}: {count} prototypes")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.tools.rebuild_prototypes",
        description="Backfill missing pose bins and rebuild face_prototypes",
    )
    parser.add_argument(
        "--person-id",
        dest="person_id",
        default=None,
        help="Rebuild only this enrolled member (default: every enrolled member)",
    )
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
