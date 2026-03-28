"""Core face detection and recognition engine wrapping InsightFace."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: list[float]  # [x1, y1, x2, y2]
    embedding: np.ndarray  # 512-dim ArcFace embedding
    det_score: float  # detection confidence
    landmarks: np.ndarray | None = None


class FaceEngine:
    """Wraps InsightFace's FaceAnalysis for detection + embedding extraction."""

    def __init__(self) -> None:
        import insightface

        model_name = config.get("face_engine.model_name", "buffalo_l")
        model_root = config.get("face_engine.model_root", "data/models")
        ctx_id = int(config.get("face_engine.ctx_id", 0))
        det_size_cfg = config.get("face_engine.det_size", [640, 640])
        det_size = tuple(det_size_cfg) if isinstance(det_size_cfg, list) else (640, 640)
        self._det_threshold = float(config.get("face_engine.det_threshold", 0.5))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Initializing FaceAnalysis: model=%s root=%s ctx_id=%d det_size=%s", model_name, model_root, ctx_id, det_size)

        self._app = insightface.app.FaceAnalysis(
            name=model_name,
            root=model_root,
            providers=providers,
        )
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Check whether CUDA provider is actually available in this runtime.
        # self._app.models maps model names to model objects; str(obj) does not
        # expose provider names, so check via onnxruntime directly.
        try:
            import onnxruntime as ort
            self.gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            self.gpu_available = False
        logger.info("FaceEngine ready, GPU available: %s", self.gpu_available)

    def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in an image and extract embeddings.

        Args:
            image: BGR numpy array (as returned by cv2.imread).

        Returns:
            List of DetectedFace with bounding boxes and 512-dim embeddings.
        """
        faces = self._app.get(image)
        results: list[DetectedFace] = []
        for face in faces:
            score = float(face.det_score)
            if score < self._det_threshold:
                continue
            results.append(
                DetectedFace(
                    bbox=[float(x) for x in face.bbox],
                    embedding=face.normed_embedding,
                    det_score=score,
                    landmarks=face.landmark_2d_106 if hasattr(face, "landmark_2d_106") else None,
                )
            )
        return results

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        return float(np.dot(emb1, emb2))


@dataclass
class IdentifyResult:
    person_id: str
    name: str
    confidence: float  # cosine similarity to best match
    bbox: list[float]


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 string (with or without data URI prefix) to a BGR numpy array."""
    import base64

    # Strip data URI prefix if present
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from base64 data")
    return img
