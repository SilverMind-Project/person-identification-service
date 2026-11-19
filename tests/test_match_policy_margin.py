"""Top-2 margin policy inside classify_match."""

from __future__ import annotations

import pytest

from app.services.enrollment_repository import CentroidHit
from app.services.match_policy import (
    GUEST_NAME,
    UNKNOWN_PERSON_ID,
    MatchThresholds,
    classify_match,
)


def test_close_top_two_drops_recognized_to_candidate_when_below_min_margin() -> None:
    thresholds = MatchThresholds(
        recognized_min_similarity=0.40,
        candidate_min_similarity=0.25,
        min_margin=0.05,
    )
    decision = classify_match(
        [
            CentroidHit("alice", "Alice", 0.55, gallery_revision=3),
            CentroidHit("bob", "Bob", 0.53, gallery_revision=2),
        ],
        thresholds,
    )
    assert decision.recognition_state == "candidate"
    assert decision.person_id == UNKNOWN_PERSON_ID
    assert decision.name == GUEST_NAME
    assert decision.best_candidate_id == "alice"
    assert decision.second_best_candidate_id == "bob"
    assert decision.second_best_similarity == pytest.approx(0.53)
    assert decision.margin == pytest.approx(0.02)
    assert decision.gallery_revision == 3


def test_sufficient_margin_recognizes_top_identity() -> None:
    thresholds = MatchThresholds(
        recognized_min_similarity=0.40,
        candidate_min_similarity=0.25,
        min_margin=0.05,
    )
    decision = classify_match(
        [
            CentroidHit("alice", "Alice", 0.55),
            CentroidHit("bob", "Bob", 0.50),
        ],
        thresholds,
    )
    assert decision.recognition_state == "recognized"
    assert decision.person_id == "alice"
    assert decision.margin == pytest.approx(0.05)


def test_absent_second_identity_gives_margin_equal_to_top1() -> None:
    thresholds = MatchThresholds(
        recognized_min_similarity=0.40,
        candidate_min_similarity=0.25,
        min_margin=0.10,
    )
    decision = classify_match([CentroidHit("alice", "Alice", 0.55)], thresholds)
    assert decision.recognition_state == "recognized"
    assert decision.second_best_candidate_id is None
    assert decision.second_best_similarity is None
    assert decision.margin == pytest.approx(0.55)


def test_empty_gallery_has_null_margin() -> None:
    thresholds = MatchThresholds(
        recognized_min_similarity=0.40,
        candidate_min_similarity=0.25,
        min_margin=0.05,
    )
    decision = classify_match([], thresholds)
    assert decision.recognition_state == "unrecognized"
    assert decision.margin is None
    assert decision.second_best_candidate_id is None
    assert decision.second_best_similarity is None
