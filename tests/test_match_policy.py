"""classify_match: recognition-state boundaries and the empty gallery."""

from __future__ import annotations

import pytest

from app.services.enrollment_repository import CentroidHit
from app.services.match_policy import MatchDecision, MatchThresholds, classify_match

_THRESHOLDS = MatchThresholds(recognized_min_similarity=0.4, candidate_min_similarity=0.25)


def _decide(similarity: float) -> MatchDecision:
    return classify_match(
        [CentroidHit("alice", "Alice", similarity), CentroidHit("bob", "Bob", similarity - 0.1)],
        _THRESHOLDS,
    )


@pytest.mark.parametrize(
    ("similarity", "state"),
    [
        (0.95, "recognized"),
        (0.4, "recognized"),  # the recognition threshold is inclusive
        (0.3999, "candidate"),
        (0.25, "candidate"),  # the candidate threshold is inclusive
        (0.2499, "unrecognized"),
        (0.0, "unrecognized"),
    ],
)
def test_state_boundaries(similarity: float, state: str) -> None:
    decision = _decide(similarity)
    assert decision.recognition_state == state
    assert decision.similarity == similarity
    assert decision.best_candidate_id == "alice"


def test_only_a_recognized_face_is_named() -> None:
    recognized = _decide(0.5)
    assert (recognized.person_id, recognized.name) == ("alice", "Alice")
    for similarity in (0.3, 0.1):
        decision = _decide(similarity)
        assert (decision.person_id, decision.name) == ("unknown", "Guest")


def test_a_negative_similarity_is_reported_unclamped() -> None:
    decision = _decide(-0.42)
    assert decision.recognition_state == "unrecognized"
    assert decision.similarity == -0.42
    assert decision.best_candidate_id == "alice"


def test_only_the_nearest_hit_decides() -> None:
    decision = classify_match(
        [CentroidHit("bob", "Bob", 0.3), CentroidHit("alice", "Alice", 0.9)], _THRESHOLDS
    )
    # Hits arrive most similar first; the policy trusts that order.
    assert (decision.recognition_state, decision.best_candidate_id) == ("candidate", "bob")


def test_empty_gallery() -> None:
    assert classify_match([], _THRESHOLDS) == MatchDecision(
        recognition_state="unrecognized",
        person_id="unknown",
        name="Guest",
        best_candidate_id=None,
        similarity=0.0,
    )
