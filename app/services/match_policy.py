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
    """Raw cosine similarity thresholds, from ``recognition.threshold`` and
    ``recognition.unknown_threshold``."""

    recognized_min_similarity: float
    candidate_min_similarity: float


@dataclass(frozen=True)
class MatchDecision:
    recognition_state: RecognitionState
    person_id: str  # UNKNOWN_PERSON_ID unless recognized
    name: str  # GUEST_NAME unless recognized
    best_candidate_id: str | None
    similarity: float


def classify_match(hits: Sequence[CentroidHit], thresholds: MatchThresholds) -> MatchDecision:
    """Classify a face from its nearest centroids (most similar first)."""
    if not hits:
        return MatchDecision(
            recognition_state="unrecognized",
            person_id=UNKNOWN_PERSON_ID,
            name=GUEST_NAME,
            best_candidate_id=None,
            similarity=0.0,
        )
    best = hits[0]
    if best.similarity >= thresholds.recognized_min_similarity:
        return MatchDecision(
            recognition_state="recognized",
            person_id=best.person_id,
            name=best.name,
            best_candidate_id=best.person_id,
            similarity=best.similarity,
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
    )
