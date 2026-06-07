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
    yaw_deg: float = 0.0  # head pose yaw (left-right), degrees
    pitch_deg: float = 0.0  # head pose pitch (up-down), degrees
    roll_deg: float = 0.0  # head pose roll (tilt), degrees
    gender: int | None = None  # InsightFace convention: 0=female, 1=male
    age: int | None = None


@dataclass
class IdentifyResult:
    person_id: str
    name: str
    confidence: float  # cosine similarity to best match
    bbox: list[float]
    best_candidate_id: str | None = None  # nearest centroid person_id, even below threshold
    similarity: float = 0.0  # raw cosine similarity to best candidate (1 - pgvector distance)
    recognition_state: str = "recognized"  # "recognized" | "candidate" | "unrecognized"
    yaw_deg: float = 0.0  # head pose yaw from detection
    pitch_deg: float = 0.0  # head pose pitch from detection
    roll_deg: float = 0.0  # head pose roll from detection
    det_score: float = 0.0  # SCRFD detection score
