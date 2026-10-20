"""Face geometry: box size and interocular distance in source-image pixels."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.face_geometry import face_geometry


def _keypoints(left_eye: tuple[float, float], right_eye: tuple[float, float]) -> np.ndarray:
    return np.array(
        [left_eye, right_eye, (50.0, 60.0), (40.0, 75.0), (60.0, 75.0)], dtype=np.float32
    )


@pytest.mark.parametrize(
    ("left_eye", "right_eye", "expected_px"),
    [
        ((30.0, 40.0), (70.0, 40.0), 40.0),  # level eyes
        ((30.0, 40.0), (60.0, 80.0), 50.0),  # 3-4-5 triangle: tilted head
        ((12.5, 7.0), (12.5, 7.0), 0.0),  # coincident keypoints are a real (degenerate) value
    ],
)
def test_interocular_is_euclidean_eye_distance(
    left_eye: tuple[float, float], right_eye: tuple[float, float], expected_px: float
) -> None:
    geometry = face_geometry([0.0, 0.0, 100.0, 120.0], _keypoints(left_eye, right_eye))
    assert geometry.interocular_px == pytest.approx(expected_px, abs=1e-6)


def test_box_size_is_exact() -> None:
    geometry = face_geometry([10.5, 20.0, 74.5, 100.0], None)
    assert geometry.face_width_px == 64.0
    assert geometry.face_height_px == 80.0


@pytest.mark.parametrize(
    "keypoints",
    [
        None,
        np.zeros((4, 2), dtype=np.float32),
        np.zeros((5, 3), dtype=np.float32),
        _keypoints((np.nan, 40.0), (70.0, 40.0)),
        _keypoints((30.0, 40.0), (np.inf, 40.0)),
    ],
    ids=["missing", "four_points", "three_columns", "nan_eye", "inf_eye"],
)
def test_unusable_keypoints_give_none_not_zero(keypoints: np.ndarray | None) -> None:
    assert face_geometry([0.0, 0.0, 10.0, 10.0], keypoints).interocular_px is None


def test_non_finite_mouth_keypoint_does_not_affect_interocular() -> None:
    keypoints = _keypoints((30.0, 40.0), (70.0, 40.0))
    keypoints[3] = (np.nan, np.nan)
    assert face_geometry([0.0, 0.0, 10.0, 10.0], keypoints).interocular_px == pytest.approx(40.0)


def test_inverted_box_clamps_to_zero_size() -> None:
    geometry = face_geometry([50.0, 50.0, 40.0, 30.0], None)
    assert geometry.face_width_px == 0.0
    assert geometry.face_height_px == 0.0
