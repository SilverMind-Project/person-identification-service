"""Face-related dataclasses shared across services without InsightFace dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: list[float]  # [x1, y1, x2, y2]
    embedding: np.ndarray  # 512-dim ArcFace embedding
    det_score: float  # detection confidence
    landmarks: np.ndarray | None = None


@dataclass
class IdentifyResult:
    person_id: str
    name: str
    confidence: float  # cosine similarity to best match
    bbox: list[float]
