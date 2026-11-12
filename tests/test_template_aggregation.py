"""Template aggregation, the template quality rule, and the float16 embedding wire form."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from app.services.face_quality import FaceQuality
from app.services.template_aggregation import (
    EMBEDDING_DIM,
    QualityWeightedMeanAggregator,
    decode_embedding_f16,
    encode_embedding_f16,
    template_quality,
)

_AGGREGATOR = QualityWeightedMeanAggregator()


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _quality(score: float, resolution: float = 0.5, sharpness: float = 0.5) -> FaceQuality:
    return FaceQuality(
        score=score,
        resolution=resolution,
        sharpness=sharpness,
        detection=0.9,
        pitch=1.0,
        version="heuristic-v1",
    )


def test_weighted_mean_equals_the_manual_computation() -> None:
    members = np.stack([_unit(1), _unit(2) * 3.0, _unit(3)])  # one member not unit length
    weights = np.asarray([0.2, 0.5, 0.3], dtype=np.float32)

    template = _AGGREGATOR.aggregate(members, weights)

    normalized = members / np.linalg.norm(members, axis=1, keepdims=True)
    manual = (weights[:, None] * normalized).sum(axis=0)
    manual /= np.linalg.norm(manual)
    np.testing.assert_allclose(template, manual, atol=1e-6)
    assert float(np.linalg.norm(template)) == pytest.approx(1.0, abs=1e-6)
    assert _AGGREGATOR.version == "qwmean-v1"


def test_a_heavier_member_pulls_the_template_toward_itself() -> None:
    a, b = _unit(1), _unit(2)
    toward_a = _AGGREGATOR.aggregate(np.stack([a, b]), np.asarray([0.9, 0.1], dtype=np.float32))
    assert float(toward_a @ a) > float(toward_a @ b)


@pytest.mark.parametrize(
    ("embeddings", "weights", "message"),
    [
        (np.zeros((0, EMBEDDING_DIM)), np.zeros(0), "at least one member"),
        (np.stack([_unit(1), _unit(2)]), np.asarray([0.5]), "one weight per member"),
        (np.stack([_unit(1), _unit(2)]), np.asarray([0.0, 0.0]), "positive weight"),
        (np.stack([_unit(1), _unit(2)]), np.asarray([0.5, -0.1]), "must not be negative"),
        (np.stack([_unit(1), np.full(EMBEDDING_DIM, np.nan)]), np.asarray([0.5, 0.5]), "finite"),
        (np.stack([_unit(1), np.zeros(EMBEDDING_DIM)]), np.asarray([0.5, 0.5]), "zero length"),
        (np.stack([_unit(1), -_unit(1)]), np.asarray([0.5, 0.5]), "cancel out"),
    ],
)
def test_invalid_members_are_rejected(
    embeddings: np.ndarray, weights: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _AGGREGATOR.aggregate(embeddings.astype(np.float32), weights.astype(np.float32))


def test_template_quality_averages_the_three_best_members_by_score() -> None:
    quality = template_quality(
        [
            _quality(0.9, resolution=0.8, sharpness=0.6),
            _quality(0.1, resolution=0.0, sharpness=0.0),  # below the top three: ignored
            _quality(0.7, resolution=0.6, sharpness=0.4),
            _quality(0.5, resolution=0.4, sharpness=0.2),
        ]
    )

    assert quality.score == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert quality.resolution == pytest.approx((0.8 + 0.6 + 0.4) / 3)
    assert quality.sharpness == pytest.approx((0.6 + 0.4 + 0.2) / 3)
    assert (quality.detection, quality.pitch) == (pytest.approx(0.9), pytest.approx(1.0))


def test_template_quality_of_two_members_uses_both() -> None:
    assert template_quality([_quality(0.2), _quality(0.6)]).score == pytest.approx(0.4)
    with pytest.raises(ValueError, match="at least one scored member"):
        template_quality([])


def test_float16_round_trip_preserves_the_embedding_to_half_precision() -> None:
    embedding = _unit(7)

    encoded = encode_embedding_f16(embedding)
    decoded = decode_embedding_f16(encoded)

    assert len(base64.b64decode(encoded)) == EMBEDDING_DIM * 2
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, embedding, atol=1e-3)
    # Little endian on the wire, whatever the host.
    assert np.frombuffer(base64.b64decode(encoded), dtype="<f2")[0] == np.float16(embedding[0])


@pytest.mark.parametrize(
    "encoded",
    [
        "not base64!",
        base64.b64encode(b"\x00" * 10).decode(),
        base64.b64encode(np.full(EMBEDDING_DIM, np.inf, dtype="<f2").tobytes()).decode(),
    ],
)
def test_bad_wire_embeddings_are_rejected(encoded: str) -> None:
    with pytest.raises(ValueError, match="embedding"):
        decode_embedding_f16(encoded)


def test_encoding_refuses_the_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="512 values"):
        encode_embedding_f16(np.zeros(128, dtype=np.float32))
