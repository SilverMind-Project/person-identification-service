"""Pure visitor-clustering policy: which sightings may enter, and which review moves are legal.

No I/O here; ``VisitorStore`` owns the SQL and asks these functions what to do.

Cluster states:

- ``candidate`` and ``surfaced`` are *open*: still unreviewed. They absorb matching sightings and
  their centroid is the running mean of everything they absorbed.
- ``dismissed`` and ``attributed`` are *closed*: an operator reviewed them. They keep absorbing
  matching sightings, so the same face does not come back as a new cluster, but their centroid is
  frozen at review time. A closed cluster that kept widening would slowly absorb faces it was
  never reviewed for.
- ``named`` belongs to an enrolled member created from it and never absorbs anything.

``attributed`` records that a cluster is faces of an existing, enrolled member. It never changes
that member's enrollment: only explicit enrollment or naming a new person does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OPEN_STATUSES: tuple[str, ...] = ("candidate", "surfaced")
CLOSED_STATUSES: tuple[str, ...] = ("dismissed", "attributed")
# Clusters a new sighting may join; also the clusters the retention purge may delete.
JOINABLE_STATUSES: tuple[str, ...] = OPEN_STATUSES + CLOSED_STATUSES

ReviewVerdict = Literal["attribute", "dismiss"]
REVIEW_VERDICTS: tuple[ReviewVerdict, ...] = ("attribute", "dismiss")

SightingRejection = Literal[
    "low_detection_score",
    "face_too_small",
    "near_member",
    "no_quality_score",
    "low_face_quality",
    "pose_too_oblique",
]

ReviewError = Literal["not_found", "already_named", "unknown_member"]


@dataclass(frozen=True)
class SightingGate:
    """Thresholds a face must pass to become a visitor sighting."""

    min_detection_score: float
    # Shortest side of the face box, in pixels of the submitted image.
    min_face_px: float
    # A face this similar to an enrolled member (raw cosine similarity) is not a visitor.
    max_member_similarity: float
    # Face quality score in [0, 1] (face_quality.py). 0.0 disables the check.
    min_face_quality: float
    # Absolute head yaw in degrees. 180 disables the check.
    max_abs_yaw_deg: float


def sighting_rejection(
    gate: SightingGate,
    *,
    det_score: float,
    face_min_side_px: float,
    member_similarity: float,
    quality_score: float | None,
    yaw_deg: float,
) -> SightingRejection | None:
    """Why a face may not become a visitor sighting, or None when it may.

    Faces an operator could not judge (blurred, small, turned away) only fragment one person
    into many clusters; the quality and pose checks keep them out. A face without a quality score
    is rejected rather than waved through.
    """
    if det_score < gate.min_detection_score:
        return "low_detection_score"
    if face_min_side_px < gate.min_face_px:
        return "face_too_small"
    # Fail closed: a borderline household match must never seed a visitor cluster.
    if member_similarity >= gate.max_member_similarity:
        return "near_member"
    if gate.min_face_quality > 0.0:
        if quality_score is None:
            return "no_quality_score"
        if quality_score < gate.min_face_quality:
            return "low_face_quality"
    if abs(yaw_deg) > gate.max_abs_yaw_deg:
        return "pose_too_oblique"
    return None


def centroid_moves_on_join(status: str) -> bool:
    """Whether a cluster's centroid follows the sightings it absorbs (open clusters only)."""
    return status in OPEN_STATUSES


def review_error(current_status: str | None, verdict: ReviewVerdict) -> ReviewError | None:
    """Why a review verdict cannot apply to a cluster in ``current_status``, or None.

    Any unreviewed or already reviewed cluster may be attributed or dismissed again, so an
    operator can correct an earlier review. A named cluster belongs to a member and is final.
    """
    if current_status is None:
        return "not_found"
    if current_status == "named":
        return "already_named"
    return None
