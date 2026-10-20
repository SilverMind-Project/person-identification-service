"""Face geometry measured in source-image pixels.

Interocular distance is taken from the two SCRFD eye keypoints because those are the points
ArcFace alignment maps onto its 112x112 template (``_prepare_arcface``): they measure the pixels
that actually reach the recognizer. Face recognition accuracy degrades sharply as interocular
distance falls toward 10 to 20 px, so this number is the primary resolution measure for a camera.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Keypoint order emitted by SCRFD: left eye, right eye, nose, left mouth corner, right mouth corner.
_LEFT_EYE = 0
_RIGHT_EYE = 1
_KEYPOINT_SHAPE = (5, 2)


@dataclass(frozen=True)
class FaceGeometry:
    """Face size in pixels of the image the face was detected in."""

    face_width_px: float
    face_height_px: float
    # None when the keypoints are missing or degenerate; never 0.0 as a sentinel.
    interocular_px: float | None


def face_geometry(bbox: Sequence[float], keypoints_5_px: np.ndarray | None) -> FaceGeometry:
    """Measure a face from its ``[x1, y1, x2, y2]`` box and its five SCRFD keypoints."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return FaceGeometry(
        face_width_px=max(0.0, x2 - x1),
        face_height_px=max(0.0, y2 - y1),
        interocular_px=_interocular_px(keypoints_5_px),
    )


def _interocular_px(keypoints_5_px: np.ndarray | None) -> float | None:
    if keypoints_5_px is None or keypoints_5_px.shape != _KEYPOINT_SHAPE:
        return None
    eyes = keypoints_5_px[[_LEFT_EYE, _RIGHT_EYE]].astype(np.float64)
    if not np.all(np.isfinite(eyes)):
        return None
    distance = float(np.linalg.norm(eyes[0] - eyes[1]))
    return distance if math.isfinite(distance) else None
