"""Core face detection and recognition engine wrapping InsightFace."""

from __future__ import annotations

import logging

import cv2
import insightface
import numpy as np

from app import config
from app.services.face_models import DetectedFace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical 3D face model (68 points) for solvePnP head-pose estimation.
# Source: OpenFace / "Facial landmark detection by deep multi-task learning"
# (Zhang et al., ECCV 2014).  The points are in a right-handed coordinate
# frame (x = right, y = down, z = away from camera) and are unit-less;
# solvePnP scales them automatically.
# ---------------------------------------------------------------------------
_CANONICAL_FACE_3D: np.ndarray = np.array(
    [
        [-0.06551, -0.16470, 0.35883],  #  0: nose tip
        [-0.08425, -0.16539, 0.33830],  #  1: right nose base
        [-0.04793, -0.17946, 0.34771],  #  2: left nose base
        [-0.10556, -0.22293, 0.27388],  #  3: right mouth top
        [-0.07039, -0.24003, 0.27681],  #  4: centre mouth top
        [-0.02272, -0.22578, 0.27853],  #  5: left mouth top
        [-0.11329, -0.25523, 0.23943],  #  6: right mouth corner
        [-0.07274, -0.26879, 0.24338],  #  7: centre mouth bottom
        [-0.01744, -0.25853, 0.24339],  #  8: left mouth corner
        [-0.15820, -0.17003, 0.18949],  #  9: right jaw 1
        [-0.12068, -0.32684, 0.20827],  # 10: right jaw 2
        [-0.07364, -0.36984, 0.21761],  # 11: chin
        [-0.01543, -0.33037, 0.21372],  # 12: left jaw 2
        [0.03492, -0.17698, 0.19435],   # 13: left jaw 1
        [-0.10127, -0.10354, 0.35388],  # 14: right eye outer
        [-0.05344, -0.08787, 0.37268],  # 15: right eye inner
        [0.00120, -0.08891, 0.37556],   # 16: between eyes (bridge)
        [0.05597, -0.09063, 0.37276],   # 17: left eye inner
        [0.10089, -0.10746, 0.34917],   # 18: left eye outer
    ],
    dtype=np.float64,
)

# InsightFace 3D 68-landmark indices that correspond to the canonical model above.
_CANONICAL_MAP_INDICES: tuple[int, ...] = (
    30,   #  0: nose tip
    31,   #  1: right nose base
    35,   #  2: left nose base
    52,   #  3: right mouth top
    62,   #  4: centre mouth top
    63,   #  5: left mouth top
    55,   #  6: right mouth corner
    66,   #  7: centre mouth bottom
    65,   #  8: left mouth corner
    1,    #  9: right jaw 1
    9,    # 10: right jaw 2
    8,    # 11: chin
    7,    # 12: left jaw 2
    15,   # 13: left jaw 1
    40,   # 14: right eye outer
    42,   # 15: right eye inner
    27,   # 16: between eyes (bridge)
    39,   # 17: left eye inner
    37,   # 18: left eye outer
)

_NUM_POSE_POINTS = len(_CANONICAL_MAP_INDICES)


class FaceEngine:
    """Wraps InsightFace's FaceAnalysis for detection + embedding extraction."""

    def __init__(self) -> None:

        model_name = config.get("face_engine.model_name")
        model_root = config.get("face_engine.model_root")
        ctx_id = int(config.get("face_engine.ctx_id"))
        det_size_cfg = config.get("face_engine.det_size", [640, 640])
        det_size = tuple(det_size_cfg) if isinstance(det_size_cfg, list) else (640, 640)
        self._det_threshold = float(config.get("face_engine.det_threshold"))

        providers = ["CUDAExecutionProvider"]
        logger.info(
            "Initializing FaceAnalysis: model=%s root=%s ctx_id=%d det_size=%s",
            model_name,
            model_root,
            ctx_id,
            det_size,
        )

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
            yaw_deg, pitch_deg, roll_deg = _estimate_head_pose(face)
            results.append(
                DetectedFace(
                    bbox=[float(x) for x in face.bbox],
                    embedding=face.normed_embedding,
                    det_score=score,
                    landmarks=face.landmark_2d_106 if hasattr(face, "landmark_2d_106") else None,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    roll_deg=roll_deg,
                )
            )
        return results

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings."""
        return float(np.dot(emb1, emb2))


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


def _estimate_head_pose(face) -> tuple[float, float, float]:
    """Estimate yaw, pitch, roll (degrees) from InsightFace 3D landmarks.

    Uses the standard solvePnP method with a canonical 19-point 3D face
    model.  When 3D landmarks are absent the result is (0, 0, 0).

    Returns:
        (yaw_deg, pitch_deg, roll_deg) where yaw < 0 is looking left,
        yaw > 0 is looking right, pitch > 0 is looking up.
    """
    try:
        lm_3d = getattr(face, "landmark_3d_68", None)
    except Exception:
        lm_3d = None

    if lm_3d is None or not hasattr(lm_3d, "shape") or lm_3d.shape[0] < 68:
        return (0.0, 0.0, 0.0)

    # Build the 2D-3D point correspondence.
    img_pts: list[list[float]] = []
    obj_pts: list[list[float]] = []
    for ci, li in enumerate(_CANONICAL_MAP_INDICES):
        obj_pts.append(_CANONICAL_FACE_3D[ci].tolist())
        pt = lm_3d[li]
        img_pts.append([float(pt[0]), float(pt[1])])

    if len(img_pts) < 6:
        return (0.0, 0.0, 0.0)

    obj_arr = np.array(obj_pts, dtype=np.float64)
    img_arr = np.array(img_pts, dtype=np.float64)

    # Use the full frame as the camera matrix: focal = frame width, centre = centre.
    # solvePnP is robust to the exact focal length; what matters is the ratio.
    focal = 640.0
    centre = 320.0
    camera_matrix = np.array(
        [[focal, 0.0, centre], [0.0, focal, centre], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _tvec = cv2.solvePnP(
        obj_arr, img_arr, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return (0.0, 0.0, 0.0)

    # Convert rotation vector to Euler angles.
    rmat, _jac = cv2.Rodrigues(rvec)
    # Decompose rotation matrix: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch_rad = np.arctan2(-rmat[2, 0], sy)
        yaw_rad = np.arctan2(rmat[1, 0], rmat[0, 0])
        roll_rad = np.arctan2(rmat[2, 1], rmat[2, 2])
    else:
        pitch_rad = np.arctan2(-rmat[2, 0], sy)
        yaw_rad = np.arctan2(-rmat[0, 1], rmat[1, 1])
        roll_rad = 0.0

    yaw_deg = float(np.degrees(yaw_rad))
    pitch_deg = float(np.degrees(pitch_rad))
    roll_deg = float(np.degrees(roll_rad))

    return (yaw_deg, pitch_deg, roll_deg)
