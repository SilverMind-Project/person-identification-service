"""Visitor clustering policy: the sighting gate and review transitions, no database."""

from __future__ import annotations

import pytest

from app.services.visitor_policy import (
    SightingGate,
    centroid_moves_on_join,
    review_error,
    sighting_rejection,
)

_GATE = SightingGate(
    min_detection_score=0.6,
    min_face_px=64.0,
    max_member_similarity=0.35,
    min_face_quality=0.7,
    max_abs_yaw_deg=30.0,
)
_PASS = {
    "det_score": 0.9,
    "face_min_side_px": 80.0,
    "member_similarity": 0.1,
    "quality_score": 0.8,
    "yaw_deg": 0.0,
}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({}, None),
        ({"det_score": 0.59}, "low_detection_score"),
        ({"face_min_side_px": 63.9}, "face_too_small"),
        ({"member_similarity": 0.35}, "near_member"),
        ({"quality_score": None}, "no_quality_score"),
        ({"quality_score": 0.69}, "low_face_quality"),
        ({"quality_score": 0.7}, None),
        ({"yaw_deg": -30.0}, None),
        ({"yaw_deg": 30.1}, "pose_too_oblique"),
        ({"yaw_deg": -45.0}, "pose_too_oblique"),
    ],
)
def test_sighting_gate(change: dict, reason: str | None) -> None:
    assert sighting_rejection(_GATE, **{**_PASS, **change}) == reason


def test_disabled_quality_and_pose_checks_accept_an_unscored_turned_face() -> None:
    off = SightingGate(0.6, 64.0, 0.35, min_face_quality=0.0, max_abs_yaw_deg=180.0)
    assert sighting_rejection(off, **{**_PASS, "quality_score": None, "yaw_deg": 90.0}) is None


@pytest.mark.parametrize(
    ("status", "moves"),
    [("candidate", True), ("surfaced", True), ("dismissed", False), ("attributed", False)],
)
def test_only_open_clusters_follow_their_sightings(status: str, moves: bool) -> None:
    assert centroid_moves_on_join(status) is moves


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (None, "not_found"),
        ("named", "already_named"),
        ("candidate", None),
        ("surfaced", None),
        ("dismissed", None),
        ("attributed", None),
    ],
)
@pytest.mark.parametrize("verdict", ["attribute", "dismiss"])
def test_review_transitions(status: str | None, error: str | None, verdict: str) -> None:
    assert review_error(status, verdict) == error  # type: ignore[arg-type]
