"""HeuristicFaceQualityScorer: component ramps, geometric-mean combination, and validation."""

from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
import pytest

from app.services.face_quality import (
    QUALITY_VERSION_HEURISTIC_V1,
    FaceQualityInputs,
    HeuristicFaceQualityScorer,
    HeuristicQualityParams,
    build_face_quality_scorer,
    laplacian_variance,
)

_PARAMS = HeuristicQualityParams()
_SCORER = HeuristicFaceQualityScorer(_PARAMS)


def _checkerboard(square_px: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:112, 0:112]
    return (((yy // square_px + xx // square_px) % 2) * 255).astype(np.uint8)


_SHARP = _checkerboard()
_FLAT = np.full((112, 112), 128, dtype=np.uint8)


def _inputs(
    interocular_px: float | None = 36.0,
    det_score: float = 0.9,
    pitch_deg: float = 0.0,
    image: np.ndarray = _SHARP,
) -> FaceQualityInputs:
    return FaceQualityInputs(
        interocular_px=interocular_px,
        det_score=det_score,
        pitch_deg=pitch_deg,
        aligned_gray_112=image,
    )


@pytest.mark.parametrize(
    ("interocular_px", "expected"),
    [(10.0, 0.0), (23.0, 0.5), (36.0, 1.0), (4.0, 0.0), (80.0, 1.0), (None, 0.0)],
)
def test_resolution_component(interocular_px: float | None, expected: float) -> None:
    assert _SCORER.score(_inputs(interocular_px=interocular_px)).resolution == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("det_score", "expected"), [(0.5, 0.0), (0.7, 0.5), (0.9, 1.0), (0.3, 0.0)]
)
def test_detection_component(det_score: float, expected: float) -> None:
    assert _SCORER.score(_inputs(det_score=det_score)).detection == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pitch_deg", "expected"),
    [(0.0, 1.0), (15.0, 1.0), (30.0, 0.5), (45.0, 0.0), (-30.0, 0.5), (-60.0, 0.0)],
)
def test_pitch_component_is_symmetric_and_descending(pitch_deg: float, expected: float) -> None:
    assert _SCORER.score(_inputs(pitch_deg=pitch_deg)).pitch == pytest.approx(expected)


def test_sharpness_component_at_the_anchors() -> None:
    # A uniform crop has zero Laplacian variance (score 0); a hard-edged checkerboard saturates.
    assert _SCORER.score(_inputs(image=_FLAT)).sharpness == 0.0
    assert _SCORER.score(_inputs(image=_SHARP)).sharpness == 1.0


def test_sharpness_midpoint_follows_the_log_ramp() -> None:
    blurred = cv2.GaussianBlur(_SHARP, (0, 0), sigmaX=2.5)
    lap_var = laplacian_variance(blurred)
    expected = (math.log10(lap_var) - _PARAMS.log_lap_var_low) / (
        _PARAMS.log_lap_var_high - _PARAMS.log_lap_var_low
    )
    assert 0.0 < expected < 1.0, "fixture must land inside the ramp"
    assert _SCORER.score(_inputs(image=blurred)).sharpness == pytest.approx(expected)


def test_blurred_copy_scores_lower_sharpness_than_the_sharp_original() -> None:
    blurred = cv2.GaussianBlur(_SHARP, (0, 0), sigmaX=3.0)
    assert (
        _SCORER.score(_inputs(image=blurred)).sharpness
        < _SCORER.score(_inputs(image=_SHARP)).sharpness
    )


def test_perfect_face_scores_one_and_carries_the_version() -> None:
    quality = _SCORER.score(_inputs())
    assert quality.score == pytest.approx(1.0)
    assert quality.version == QUALITY_VERSION_HEURISTIC_V1 == _SCORER.version


def test_combined_score_is_the_weighted_geometric_mean() -> None:
    quality = _SCORER.score(_inputs(interocular_px=23.0, det_score=0.7, pitch_deg=30.0))
    expected = 0.5**0.45 * 1.0**0.25 * 0.5**0.15 * 0.5**0.15
    assert quality.score == pytest.approx(expected)


def test_a_zero_component_is_floored_not_fatal() -> None:
    zero_resolution = _SCORER.score(_inputs(interocular_px=None)).score
    low_resolution = _SCORER.score(_inputs(interocular_px=12.0)).score  # component 2/26 > floor
    assert zero_resolution == pytest.approx(_PARAMS.component_floor**_PARAMS.weight_resolution)
    assert 0.0 < zero_resolution < low_resolution


def test_a_tiny_sharp_frontal_face_scores_low() -> None:
    # The reason for a geometric mean: an arithmetic mean would give this face about 0.55.
    assert _SCORER.score(_inputs(interocular_px=8.0)).score < 0.2


def test_scoring_is_deterministic() -> None:
    inputs = _inputs(interocular_px=19.0, det_score=0.66, pitch_deg=22.0)
    assert _SCORER.score(inputs) == _SCORER.score(inputs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_resolution": 0.5},
        {"weight_pitch": -0.15, "weight_resolution": 0.75},
        {"component_floor": 0.0},
        {"iod_low_px": 36.0},
    ],
)
def test_invalid_params_are_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="face quality"):
        HeuristicFaceQualityScorer(replace(_PARAMS, **overrides))


def test_factory_builds_the_heuristic_scorer_from_settings() -> None:
    scorer = build_face_quality_scorer({"scorer": "heuristic-v1", "heuristic": {"iod_high_px": 40}})
    assert scorer.version == QUALITY_VERSION_HEURISTIC_V1
    assert scorer.score(_inputs(interocular_px=38.0)).resolution == pytest.approx(28.0 / 30.0)


@pytest.mark.parametrize(
    "section",
    [{}, {"scorer": "magface"}, {"scorer": "heuristic-v1", "heuristic": {"iod_low": 10}}],
)
def test_factory_rejects_unknown_scorers_and_keys(section: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="face_quality"):
        build_face_quality_scorer(section)
