"""Pure prototype construction, pose binning, quality weighting, and promotion eligibility."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.prototype_builder import (
    MemberEmbedding,
    PrototypePolicy,
    build_prototypes,
    effective_quality_weight,
    pose_bin,
    promotion_ineligibility,
)


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.zeros(512, dtype=np.float32)
    arr[: len(vec)] = vec
    return (arr / np.linalg.norm(arr)).astype(np.float32)


@pytest.mark.parametrize(
    ("yaw", "pitch", "expected"),
    [
        (0.0, 0.0, "frontal"),
        (19.9, 19.9, "frontal"),
        (-19.9, -19.9, "frontal"),
        (-20.0, 0.0, "left"),
        (-45.0, 19.9, "left"),
        (20.0, 0.0, "right"),
        (35.0, -19.9, "right"),
        # Down checks abs(pitch) >= 20 first, overriding yaw (matches OpenCV ceiling-camera +pitch).
        (0.0, 20.0, "down"),
        (0.0, -20.0, "down"),
        (45.0, 25.0, "down"),
        (-45.0, -25.0, "down"),
    ],
)
def test_pose_bin_boundaries(yaw: float, pitch: float, expected: str) -> None:
    assert pose_bin(yaw, pitch) == expected


def test_small_bin_keeps_individual_prototypes_and_respects_max_per_bin() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    items = [
        MemberEmbedding(
            embedding=e1,
            embedding_id=10,
            pose_bin="left",
            quality_score=0.4,
            quality_version="heuristic-v1",
        ),
        MemberEmbedding(
            embedding=e2,
            embedding_id=11,
            pose_bin="left",
            quality_score=0.9,
            quality_version="heuristic-v1",
        ),
    ]
    protos = build_prototypes("alice", items, PrototypePolicy(min_members_per_bin=3, max_per_bin=6))
    assert len(protos) == 2
    assert all(p.pose_bin == "left" and p.member_count == 1 for p in protos)
    # Ranked highest quality first.
    assert [p.source_embedding_id for p in protos] == [11, 10]

    # When min_members_per_bin > max_per_bin, individual samples cap at max_per_bin highest quality.
    capped = build_prototypes("alice", items, PrototypePolicy(min_members_per_bin=5, max_per_bin=1))
    assert len(capped) == 1
    assert capped[0].source_embedding_id == 11


def test_bin_at_or_above_min_members_collapses_into_quality_weighted_mean() -> None:
    low_a = _unit([1.0, 0.0])
    low_b = _unit([1.0, 0.0])
    high = _unit([0.0, 1.0])
    policy = PrototypePolicy(min_members_per_bin=3, missing_quality_weight=0.5)
    items = [
        MemberEmbedding(
            embedding=low_a,
            embedding_id=1,
            pose_bin="frontal",
            quality_score=0.1,
            quality_version="heuristic-v1",
        ),
        MemberEmbedding(
            embedding=low_b,
            embedding_id=2,
            pose_bin="frontal",
            quality_score=0.1,
            quality_version="heuristic-v1",
        ),
        MemberEmbedding(
            embedding=high,
            embedding_id=3,
            pose_bin="frontal",
            quality_score=0.9,
            quality_version="heuristic-v1",
            crop_object_key="calibration/high.png",
        ),
    ]
    protos = build_prototypes("alice", items, policy)
    assert len(protos) == 1
    p = protos[0]
    assert p.member_count == 3
    assert p.source_embedding_id is None
    assert p.crop_object_key == "calibration/high.png"
    assert p.quality_version == "heuristic-v1"
    # High-quality axis dominates over the two low-quality embeddings (0.9 vs 0.2).
    assert p.embedding[1] > p.embedding[0]
    assert np.linalg.norm(p.embedding) == pytest.approx(1.0, abs=1e-6)


def test_missing_or_mismatched_quality_version_uses_conservative_default() -> None:
    policy = PrototypePolicy(missing_quality_weight=0.5, expected_quality_version="heuristic-v1")
    assert effective_quality_weight(None, None, policy) == 0.5
    assert effective_quality_weight(0.95, "other-v2", policy) == 0.5
    assert effective_quality_weight(0.82, "heuristic-v1", policy) == pytest.approx(0.82)

    # Pooled bin with only unscored embeddings keeps quality_score/quality_version None.
    items = [MemberEmbedding(embedding=_unit([1.0, 0.0]), embedding_id=i) for i in range(3)]
    protos = build_prototypes("alice", items, policy)
    assert len(protos) == 1
    assert protos[0].pose_bin == "frontal"
    assert protos[0].quality_score is None
    assert protos[0].quality_version is None


def test_build_prototypes_is_deterministic() -> None:
    items = [
        MemberEmbedding(
            embedding=_unit([float(i + 1), float(5 - i)]),
            embedding_id=i,
            pose_bin="down" if i % 2 else "frontal",
            quality_score=0.2 * (i + 1),
            quality_version="heuristic-v1",
        )
        for i in range(4)
    ]
    first = build_prototypes("alice", items)
    second = build_prototypes("alice", items)
    assert [(p.prototype_id, p.pose_bin, p.member_count) for p in first] == [
        (p.prototype_id, p.pose_bin, p.member_count) for p in second
    ]
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a.embedding, b.embedding, atol=1e-7)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, None),
        ({"label_state": "unlabeled"}, "not_labeled"),
        ({"label_kind": "not_enrolled", "label_identity_id": None}, "not_member_label"),
        ({"already_promoted": True}, "already_promoted"),
        ({"quality_version": "heuristic-v2"}, "quality_version_mismatch"),
        ({"quality_score": 0.49}, "quality_too_low"),
        ({"interocular_px": 23.9}, "interocular_too_small"),
        ({"interocular_px": None}, "interocular_too_small"),
    ],
)
def test_promotion_ineligibility_gates(overrides: dict[str, object], expected: str | None) -> None:
    kwargs: dict[str, object] = {
        "label_state": "labeled",
        "label_kind": "member",
        "label_identity_id": "alice",
        "quality_score": 0.65,
        "quality_version": "heuristic-v1",
        "interocular_px": 28.0,
        "active_quality_version": "heuristic-v1",
        "policy": PrototypePolicy(min_promotion_quality=0.5, min_promotion_interocular_px=24.0),
        "already_promoted": False,
    }
    kwargs.update(overrides)
    assert promotion_ineligibility(**kwargs) == expected  # type: ignore[arg-type]
