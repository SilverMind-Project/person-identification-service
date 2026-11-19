"""Pure multi-prototype construction, pose binning, and promotion eligibility policy.

Database-free and config-free so every boundary (pose angles, quality weighting, bin caps,
promotion floors) is unit-tested deterministically.

Pose bins match the head-pose convention of ``app.services.face_engine`` (OpenCV camera frame:
``+X`` right, ``+Y`` down, ``+Z`` into scene; ceiling cameras looking down produce positive
``pitch_deg``): ``down`` is checked first as ``abs(pitch_deg) >= 20`` (matching steep pitch in
calibration analysis), and ``frontal``/``left``/``right`` partition yaw within ``abs(pitch) < 20``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

PoseBin = Literal["frontal", "left", "right", "down"]
POSE_BINS: tuple[PoseBin, ...] = ("frontal", "left", "right", "down")

EmbeddingSource = Literal[
    "enrollment_photo",
    "deployment_capture",
    "visitor_naming",
    "wearable_exemplar",
]

# Fixed namespace for deterministic prototype UUIDs across rebuilds.
_PROTOTYPE_NAMESPACE = uuid.UUID("7f3b5c1a-4b64-4d3d-9b3e-8d1c2e6b4a10")
_MIN_WEIGHT = 1e-6


@dataclass(frozen=True)
class PrototypePolicy:
    """Knobs controlling pose-bin prototype construction and candidate promotion."""

    min_members_per_bin: int = 3
    max_per_bin: int = 6
    missing_quality_weight: float = 0.5
    expected_quality_version: str = "heuristic-v1"
    min_promotion_quality: float = 0.5
    min_promotion_interocular_px: float = 24.0


@dataclass(frozen=True)
class MemberEmbedding:
    """One stored embedding for an enrolled member with pose and provenance metadata."""

    embedding: np.ndarray = field(repr=False, compare=False)
    embedding_id: int | None = None
    pose_bin: PoseBin | None = None
    source: EmbeddingSource = "enrollment_photo"
    quality_score: float | None = None
    quality_version: str | None = None
    source_candidate_id: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    crop_object_key: str | None = None


@dataclass(frozen=True)
class Prototype:
    """One pose-binned prototype representing one or more embeddings of a member."""

    prototype_id: str
    person_id: str
    pose_bin: PoseBin
    embedding: np.ndarray = field(repr=False, compare=False)
    member_count: int
    source: EmbeddingSource
    quality_score: float | None = None
    quality_version: str | None = None
    crop_object_key: str | None = None
    source_embedding_id: int | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    created_at: datetime | None = None


def pose_bin(yaw_deg: float, pitch_deg: float) -> PoseBin:
    """Map head yaw and pitch (degrees) to one of the four canonical pose bins."""
    if abs(pitch_deg) >= 20.0:
        return "down"
    if yaw_deg <= -20.0:
        return "left"
    if yaw_deg >= 20.0:
        return "right"
    return "frontal"


def effective_quality_weight(
    quality_score: float | None,
    quality_version: str | None,
    policy: PrototypePolicy,
) -> float:
    """Comparable weight for one embedding.

    Quality scores are comparable only within one scorer version. Unscored embeddings or scores
    from another version receive ``policy.missing_quality_weight`` (conservative default 0.5).
    """
    if quality_score is not None and quality_version == policy.expected_quality_version:
        return max(float(quality_score), _MIN_WEIGHT)
    return max(float(policy.missing_quality_weight), _MIN_WEIGHT)


def _unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr.astype(np.float32)
    return (arr / norm).astype(np.float32)


def _prototype_uuid(key: str) -> str:
    return str(uuid.uuid5(_PROTOTYPE_NAMESPACE, key))


def build_prototypes(
    person_id: str,
    embeddings: Sequence[MemberEmbedding],
    policy: PrototypePolicy = PrototypePolicy(),
) -> list[Prototype]:
    """Build deterministic pose-binned prototypes for one enrolled member.

    Bins with at least ``policy.min_members_per_bin`` embeddings collapse into one quality-weighted
    normalized mean prototype. Smaller bins keep each embedding as its own prototype, capped at
    ``policy.max_per_bin`` (keeping the highest-quality samples first).
    """
    grouped: dict[PoseBin, list[tuple[int, MemberEmbedding]]] = {b: [] for b in POSE_BINS}
    for idx, item in enumerate(embeddings):
        target_bin: PoseBin = item.pose_bin if item.pose_bin is not None else "frontal"
        grouped[target_bin].append((idx, item))

    prototypes: list[Prototype] = []
    for bin_name in POSE_BINS:
        bucket = grouped[bin_name]
        if not bucket:
            continue

        weighted = [
            (
                effective_quality_weight(item.quality_score, item.quality_version, policy),
                idx,
                item,
            )
            for idx, item in bucket
        ]

        if len(weighted) >= policy.min_members_per_bin:
            vectors = np.stack(
                [_unit(item.embedding).astype(np.float64) for _, _, item in weighted]
            )
            weights = np.asarray([w for w, _, _ in weighted], dtype=np.float64)
            mean_vec = np.average(vectors, axis=0, weights=weights)
            proto_vec = _unit(mean_vec)

            comparable = [
                (w, item.quality_score)
                for w, _, item in weighted
                if item.quality_score is not None
                and item.quality_version == policy.expected_quality_version
            ]
            if comparable:
                comp_weights = np.asarray([w for w, _ in comparable], dtype=np.float64)
                comp_scores = np.asarray([s for _, s in comparable], dtype=np.float64)
                pooled_quality: float | None = float(np.average(comp_scores, weights=comp_weights))
                pooled_version: str | None = policy.expected_quality_version
            else:
                pooled_quality = None
                pooled_version = None

            # Pick the highest-weight member for provenance metadata, preferring one with a crop.
            ranked = sorted(
                weighted,
                key=lambda entry: (-entry[0], 0 if entry[2].crop_object_key else 1, entry[1]),
            )
            rep = ranked[0][2]
            crop_key = next(
                (entry[2].crop_object_key for entry in ranked if entry[2].crop_object_key),
                None,
            )
            approved_entry = next(
                (entry[2] for entry in ranked if entry[2].approved_by is not None),
                rep,
            )
            prototypes.append(
                Prototype(
                    prototype_id=_prototype_uuid(f"{person_id}:{bin_name}:pooled"),
                    person_id=person_id,
                    pose_bin=bin_name,
                    embedding=proto_vec,
                    member_count=len(weighted),
                    source=rep.source,
                    quality_score=pooled_quality,
                    quality_version=pooled_version,
                    crop_object_key=crop_key,
                    source_embedding_id=None,
                    approved_by=approved_entry.approved_by,
                    approved_at=approved_entry.approved_at,
                )
            )
            continue

        # Small bin: keep individual samples, capped at max_per_bin highest quality.
        ranked = sorted(weighted, key=lambda entry: (-entry[0], entry[1]))
        kept = ranked[: policy.max_per_bin]
        for rank, (_, orig_idx, item) in enumerate(kept):
            sample_key = (
                f"id:{item.embedding_id}"
                if item.embedding_id is not None
                else f"idx:{orig_idx}:{rank}"
            )
            prototypes.append(
                Prototype(
                    prototype_id=_prototype_uuid(f"{person_id}:{bin_name}:sample:{sample_key}"),
                    person_id=person_id,
                    pose_bin=bin_name,
                    embedding=_unit(item.embedding),
                    member_count=1,
                    source=item.source,
                    quality_score=item.quality_score,
                    quality_version=item.quality_version,
                    crop_object_key=item.crop_object_key,
                    source_embedding_id=item.embedding_id,
                    approved_by=item.approved_by,
                    approved_at=item.approved_at,
                )
            )

    return prototypes


def promotion_ineligibility(
    *,
    label_state: str,
    label_kind: str | None,
    label_identity_id: str | None,
    quality_score: float | None,
    quality_version: str | None,
    interocular_px: float | None,
    active_quality_version: str,
    policy: PrototypePolicy,
    already_promoted: bool = False,
) -> str | None:
    """Return a bounded reason code when a capture cannot be promoted, or None when eligible."""
    if label_state != "labeled":
        return "not_labeled"
    if label_kind != "member" or not label_identity_id:
        return "not_member_label"
    if already_promoted:
        return "already_promoted"
    if quality_score is None or quality_version != active_quality_version:
        return "quality_version_mismatch"
    if quality_score < policy.min_promotion_quality:
        return "quality_too_low"
    if interocular_px is None or interocular_px < policy.min_promotion_interocular_px:
        return "interocular_too_small"
    return None
