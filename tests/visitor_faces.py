"""Faces for visitor-clustering tests.

The visitor gate rejects a face without a quality score or below the configured quality, so test
faces carry a realistic score unless a test is about the gate itself.
"""

from __future__ import annotations

import numpy as np

from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality


def face_quality(score: float) -> FaceQuality:
    return FaceQuality(
        score=score, resolution=score, sharpness=score, detection=score, pitch=1.0, version="test"
    )


GOOD_QUALITY = face_quality(0.9)


def visitor_face(
    embedding: np.ndarray,
    *,
    det_score: float = 0.95,
    bbox: list[float] | None = None,
    quality: FaceQuality | None = GOOD_QUALITY,
    yaw_deg: float = 0.0,
) -> DetectedFace:
    return DetectedFace(
        bbox=bbox or [10.0, 20.0, 110.0, 140.0],
        embedding=embedding,
        det_score=det_score,
        quality=quality,
        yaw_deg=yaw_deg,
    )
