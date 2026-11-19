"""Matching policy: how a face's nearest enrolled centroids become a recognition state.

Pure and database-free, so the boundary semantics are unit-tested exactly. The three states:

- ``recognized``: the nearest similarity is at or above ``recognized_min_similarity``; the face
  is named.
- ``candidate``: at or above ``candidate_min_similarity`` but below the recognition threshold; the
  face stays ``"unknown"`` but carries its nearest member as ``best_candidate_id``.
- ``unrecognized``: below both, or an empty gallery.

``best_candidate_id`` names the nearest member in every state except the empty gallery. Raw
similarity is reported as measured (it can be negative); it is not a probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.services.enrollment_repository import CentroidHit

RecognitionState = Literal["recognized", "candidate", "unrecognized"]

UNKNOWN_PERSON_ID = "unknown"
GUEST_NAME = "Guest"


@dataclass(frozen=True)
class MatchThresholds:
    """Raw cosine similarity thresholds and minimum top-2 margin."""

    recognized_min_similarity: float
    candidate_min_similarity: float
    min_margin: float = 0.0


@dataclass(frozen=True)
class MatchDecision:
    recognition_state: RecognitionState
    person_id: str  # UNKNOWN_PERSON_ID unless recognized
    name: str  # GUEST_NAME unless recognized
    best_candidate_id: str | None
    similarity: float
    second_best_candidate_id: str | None = None
    second_best_similarity: float | None = None
    margin: float | None = None
    gallery_revision: int = 1


def classify_match(hits: Sequence[CentroidHit], thresholds: MatchThresholds) -> MatchDecision:
    """Classify a face from its nearest identities (most similar first).

    ``recognized`` requires both ``top1 >= recognized_min_similarity`` and
    ``margin >= min_margin`` where ``margin = top1 - top2`` (or ``top1`` when no second identity
    exists). A face close to two enrolled people drops to ``candidate`` rather than ``recognized``.
    """
    if not hits:
        return MatchDecision(
            recognition_state="unrecognized",
            person_id=UNKNOWN_PERSON_ID,
            name=GUEST_NAME,
            best_candidate_id=None,
            similarity=0.0,
            second_best_candidate_id=None,
            second_best_similarity=None,
            margin=None,
            gallery_revision=1,
        )
    best = hits[0]
    second = hits[1] if len(hits) > 1 else None
    margin = best.similarity - second.similarity if second is not None else best.similarity
    second_id = second.person_id if second is not None else None
    second_sim = second.similarity if second is not None else None

    if best.similarity >= thresholds.recognized_min_similarity and margin >= thresholds.min_margin:
        return MatchDecision(
            recognition_state="recognized",
            person_id=best.person_id,
            name=best.name,
            best_candidate_id=best.person_id,
            similarity=best.similarity,
            second_best_candidate_id=second_id,
            second_best_similarity=second_sim,
            margin=margin,
            gallery_revision=best.gallery_revision,
        )
    state: RecognitionState = (
        "candidate" if best.similarity >= thresholds.candidate_min_similarity else "unrecognized"
    )
    return MatchDecision(
        recognition_state=state,
        person_id=UNKNOWN_PERSON_ID,
        name=GUEST_NAME,
        best_candidate_id=best.person_id,
        similarity=best.similarity,
        second_best_candidate_id=second_id,
        second_best_similarity=second_sim,
        margin=margin,
        gallery_revision=best.gallery_revision,
    )
