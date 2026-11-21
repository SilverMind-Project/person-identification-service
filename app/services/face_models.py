"""Face-related dataclasses shared across services without InsightFace dependency."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.services.face_geometry import FaceGeometry, face_geometry
from app.services.face_quality import FaceQuality


@dataclass
class DetectedFace:
    bbox: list[float]  # [x1, y1, x2, y2]
    embedding: np.ndarray  # 512-dim ArcFace embedding
    det_score: float  # detection confidence
    landmarks: np.ndarray | None = None
    # Five SCRFD keypoints (left eye, right eye, nose, left mouth, right mouth) in input-image
    # pixels, shape (5, 2). These are the points ArcFace alignment uses.
    keypoints_5_px: np.ndarray | None = None
    yaw_deg: float = 0.0  # head pose yaw (left-right), degrees
    pitch_deg: float = 0.0  # head pose pitch (up-down), degrees
    roll_deg: float = 0.0  # head pose roll (tilt), degrees
    gender: int | None = None  # InsightFace convention: 0=female, 1=male
    age: int | None = None
    # Set by FaceEngine from the aligned crop; None for a face that was not scored.
    quality: FaceQuality | None = None
    # Aligned BGR crop (input_size_px x input_size_px) produced during recognition; kept in memory
    # only so enrollment can persist it without re-running alignment (excluded from repr/equality).
    aligned_crop_bgr: np.ndarray | None = field(default=None, repr=False, compare=False)
    # Derived from bbox and keypoints_5_px when the face is built, so every producer (the engine
    # and test fakes alike) carries the one measurement face_geometry() defines.
    geometry: FaceGeometry = field(init=False)

    def __post_init__(self) -> None:
        self.geometry = face_geometry(self.bbox, self.keypoints_5_px)


@dataclass
class IdentifyResult:
    person_id: str
    name: str
    bbox: list[float]
    best_candidate_id: str | None = None  # nearest centroid person_id, even below threshold
    similarity: float = 0.0  # raw cosine similarity to best candidate (1 - pgvector distance)
    recognition_state: str = "recognized"  # "recognized" | "candidate" | "unrecognized"
    yaw_deg: float = 0.0  # head pose yaw from detection
    pitch_deg: float = 0.0  # head pose pitch from detection
    roll_deg: float = 0.0  # head pose roll from detection
    det_score: float = 0.0  # SCRFD detection score
    face_width_px: float = 0.0  # face box width in input-image pixels
    face_height_px: float = 0.0  # face box height in input-image pixels
    interocular_px: float | None = None  # eye-keypoint distance in input-image pixels
    quality: FaceQuality | None = None  # face image quality, carried from DetectedFace
    second_best_candidate_id: str | None = None  # runner-up identity id when gallery has >= 2
    second_best_similarity: float | None = None  # raw cosine similarity to runner-up identity
    margin: float | None = None  # top1 - top2 similarity (or top1 when only one identity exists)
    gallery_representation: str = "centroid"  # "centroid" | "prototypes-v1"
    gallery_revision: int = 1  # matched member's live gallery revision
