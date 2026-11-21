"""Build a parallel enrollment gallery for a target recognizer profile from retained crops.

Usage:
    DATABASE_URL=postgresql://... TRITON_GRPC_URL=triton:8701 \\
        uv run python -m app.tools.reembed --profile adaface_ir50_webface4m

Reads retained 112x112 aligned PNG crops (``enrollment/<person_id>/<embedding_id>.png`` or
promoted candidate crops) from MinIO for the source profile, embeds each crop with the target
profile's Triton recognition model, and writes the resulting embeddings, centroids, and
pose-binned prototypes under the target profile's ``recognizer_version``. The source/active
profile's gallery is never modified.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, replace
from typing import Protocol

import asyncpg
import cv2
import numpy as np
from pgvector.asyncpg import register_vector
from triton_shared.client import TritonGrpcClient

from app import config
from app.services.calibration_capture import MinioCropStore
from app.services.enrollment_repository import EnrollmentRepository, PostgresEnrollmentRepository
from app.services.face_engine import FaceEngine
from app.services.face_quality import build_face_quality_scorer
from app.services.minio_client import create_minio_client
from app.services.prototype_builder import MemberEmbedding, PrototypePolicy
from app.services.recognizer_profile import (
    DEFAULT_EMBEDDING_DIM,
    RecognizerProfile,
    get_recognizer_profile,
    load_active_recognizer_profile,
)


class AlignedCropEmbedder(Protocol):
    async def embed_aligned_crop(self, aligned_bgr: np.ndarray) -> np.ndarray: ...


class CropReader(Protocol):
    async def get_bytes(self, key: str) -> bytes | None: ...


@dataclass(frozen=True)
class MemberReembedOutcome:
    person_id: str
    source_embeddings: int
    reembedded: int
    skipped_missing_crop: int
    skipped_unreadable_crop: int


@dataclass(frozen=True)
class ReembedSummary:
    source_profile: str
    source_recognizer_version: str
    target_profile: str
    target_recognizer_version: str
    members: tuple[MemberReembedOutcome, ...]
    dry_run: bool = False

    @property
    def total_reembedded(self) -> int:
        return sum(m.reembedded for m in self.members)

    @property
    def total_skipped(self) -> int:
        return sum(m.skipped_missing_crop + m.skipped_unreadable_crop for m in self.members)


def validate_reembed_profiles(
    source_profile: RecognizerProfile,
    target_profile: RecognizerProfile,
) -> None:
    """Fail fast when profiles cannot share the 512-dim tables or retained aligned crops."""
    if source_profile.recognizer_version == target_profile.recognizer_version:
        raise ValueError(
            "Target recognizer_version must differ from source recognizer_version "
            f"({target_profile.recognizer_version!r}); refusing to overwrite the active gallery"
        )
    if target_profile.embedding_dim != DEFAULT_EMBEDDING_DIM:
        raise ValueError(
            f"Profile {target_profile.name!r} has embedding_dim={target_profile.embedding_dim}; "
            f"shared gallery tables hold vector({DEFAULT_EMBEDDING_DIM}) and require a dedicated "
            "migration with profile-scoped tables for other dimensions"
        )
    if not target_profile.templates_compatible(source_profile):
        raise ValueError(
            f"Profile {target_profile.name!r} uses a different alignment template or input size "
            f"than {source_profile.name!r}; retained aligned crops cannot be re-embedded directly"
        )


def _decode_png_bgr(raw: bytes, *, expected_size_px: int) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    if decoded.ndim != 3 or decoded.shape[:2] != (expected_size_px, expected_size_px):
        return None
    return decoded


async def reembed_gallery(
    *,
    source_repo: EnrollmentRepository,
    target_repo: EnrollmentRepository,
    crop_reader: CropReader,
    embedder: AlignedCropEmbedder,
    source_profile: RecognizerProfile,
    target_profile: RecognizerProfile,
    prototype_policy: PrototypePolicy = PrototypePolicy(),
    person_id: str | None = None,
    dry_run: bool = False,
) -> ReembedSummary:
    """Re-embed retained enrollment crops into a parallel gallery for *target_profile*.

    Writes exclusively through *target_repo* (scoped to ``target_profile.recognizer_version``)
    and never mutates *source_repo*.
    """
    validate_reembed_profiles(source_profile, target_profile)
    if source_repo.recognizer_version != source_profile.recognizer_version:
        raise ValueError("source_repo.recognizer_version does not match source_profile")
    if target_repo.recognizer_version != target_profile.recognizer_version:
        raise ValueError("target_repo.recognizer_version does not match target_profile")

    source_records = await source_repo.member_embedding_records()
    if person_id is not None:
        if not await source_repo.member_exists(person_id):
            raise LookupError(f"member '{person_id}' not found")
        target_ids = [person_id]
    else:
        target_ids = [m.person_id for m in await source_repo.list_members()]

    outcomes: list[MemberReembedOutcome] = []
    for pid in target_ids:
        items = source_records.get(pid, [])
        rebuilt: list[MemberEmbedding] = []
        missing_crop = 0
        unreadable_crop = 0
        for item in items:
            if not item.crop_object_key:
                missing_crop += 1
                continue
            raw = await crop_reader.get_bytes(item.crop_object_key)
            if raw is None:
                missing_crop += 1
                continue
            crop = _decode_png_bgr(raw, expected_size_px=target_profile.input_size_px)
            if crop is None:
                unreadable_crop += 1
                continue
            vector = await embedder.embed_aligned_crop(crop)
            rebuilt.append(
                replace(
                    item,
                    embedding=vector,
                    embedding_id=None,
                    # Quality scores from the source recognizer do not carry over across profiles;
                    # leave them null so prototypes weigh conservatively until re-scored.
                    quality_score=None,
                    quality_version=None,
                    recognizer_version=target_profile.recognizer_version,
                )
            )
        if not dry_run and rebuilt:
            await target_repo.replace_profile_embeddings(pid, rebuilt, policy=prototype_policy)
        outcomes.append(
            MemberReembedOutcome(
                person_id=pid,
                source_embeddings=len(items),
                reembedded=len(rebuilt),
                skipped_missing_crop=missing_crop,
                skipped_unreadable_crop=unreadable_crop,
            )
        )

    return ReembedSummary(
        source_profile=source_profile.name,
        source_recognizer_version=source_profile.recognizer_version,
        target_profile=target_profile.name,
        target_recognizer_version=target_profile.recognizer_version,
        members=tuple(outcomes),
        dry_run=dry_run,
    )


async def _run(args: argparse.Namespace) -> int:
    dsn = config.get("database.dsn")
    if not dsn:
        print("FAIL: database.dsn is not configured (set DATABASE_URL)")
        return 1

    face_cfg = config.get("face_engine", {}) or {}
    source_profile = (
        get_recognizer_profile(face_cfg, args.source_profile)
        if args.source_profile
        else load_active_recognizer_profile(face_cfg)
    )
    target_profile = get_recognizer_profile(face_cfg, args.profile)
    validate_reembed_profiles(source_profile, target_profile)

    proto_cfg = config.get("prototypes", {}) or {}
    base_quality = str(config.get("face_quality.scorer", "heuristic-v1"))
    policy = PrototypePolicy(
        min_members_per_bin=int(proto_cfg.get("min_members_per_bin", 3)),
        max_per_bin=int(proto_cfg.get("max_per_bin", 6)),
        missing_quality_weight=float(proto_cfg.get("missing_quality_weight", 0.5)),
        expected_quality_version=target_profile.scoped_quality_version(base_quality),
        min_promotion_quality=float(proto_cfg.get("min_promotion_quality", 0.5)),
        min_promotion_interocular_px=float(proto_cfg.get("min_promotion_interocular_px", 24.0)),
    )

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(dsn, init=_init, min_size=1, max_size=2)
    minio_client = create_minio_client()
    crop_store = MinioCropStore(minio_client)
    triton_client = TritonGrpcClient(
        str(config.get("face_engine.triton_url", "triton:8701")),
        timeout_ms=int(config.get("face_engine.triton_timeout_ms", 30_000)),
    )
    scorer = build_face_quality_scorer(config.get("face_quality", {}))

    try:
        async with triton_client:
            engine = FaceEngine(triton_client, scorer, target_profile)
            await engine.validate_recognition_model()
            source_repo = PostgresEnrollmentRepository(
                pool, recognizer_version=source_profile.recognizer_version
            )
            target_repo = PostgresEnrollmentRepository(
                pool, recognizer_version=target_profile.recognizer_version
            )
            summary = await reembed_gallery(
                source_repo=source_repo,
                target_repo=target_repo,
                crop_reader=crop_store,
                embedder=engine,
                source_profile=source_profile,
                target_profile=target_profile,
                prototype_policy=policy,
                person_id=args.person_id,
                dry_run=args.dry_run,
            )
    except (LookupError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        await pool.close()

    mode = "DRY-RUN" if summary.dry_run else "OK"
    print(
        f"{mode}: source={summary.source_profile} ({summary.source_recognizer_version}) -> "
        f"target={summary.target_profile} ({summary.target_recognizer_version}) "
        f"reembedded={summary.total_reembedded} skipped={summary.total_skipped}"
    )
    for m in summary.members:
        print(
            f"  {m.person_id}: source={m.source_embeddings} reembedded={m.reembedded} "
            f"missing_crop={m.skipped_missing_crop} unreadable={m.skipped_unreadable_crop}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.tools.reembed",
        description="Build a parallel enrollment gallery for a recognizer profile from retained crops",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Target recognizer profile name from face_engine.profiles",
    )
    parser.add_argument(
        "--source-profile",
        dest="source_profile",
        default=None,
        help="Source profile holding retained crops (default: active_profile)",
    )
    parser.add_argument(
        "--person-id",
        dest="person_id",
        default=None,
        help="Re-embed only this enrolled member (default: all members)",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Run inference and report counts without writing to Postgres",
    )
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
