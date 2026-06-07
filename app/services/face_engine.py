"""Triton-backed Buffalo_L face detection and recognition."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Protocol, cast

import cv2
import numpy as np
import numpy.typing as npt
from triton_shared.client import TritonClientProtocol

from app import config
from app.services.face_models import DetectedFace

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]


class LandmarkFaceProtocol(Protocol):
    @property
    def landmark_3d_68(self) -> FloatArray | None: ...


_SCRFD_OUTPUTS = [
    "448",
    "471",
    "494",
    "451",
    "474",
    "497",
    "454",
    "477",
    "500",
]
_ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_CANONICAL_FACE_3D = np.array(
    [
        [-0.06551, -0.16470, 0.35883],
        [-0.08425, -0.16539, 0.33830],
        [-0.04793, -0.17946, 0.34771],
        [-0.10556, -0.22293, 0.27388],
        [-0.07039, -0.24003, 0.27681],
        [-0.02272, -0.22578, 0.27853],
        [-0.11329, -0.25523, 0.23943],
        [-0.07274, -0.26879, 0.24338],
        [-0.01744, -0.25853, 0.24339],
        [-0.15820, -0.17003, 0.18949],
        [-0.12068, -0.32684, 0.20827],
        [-0.07364, -0.36984, 0.21761],
        [-0.01543, -0.33037, 0.21372],
        [0.03492, -0.17698, 0.19435],
        [-0.10127, -0.10354, 0.35388],
        [-0.05344, -0.08787, 0.37268],
        [0.00120, -0.08891, 0.37556],
        [0.05597, -0.09063, 0.37276],
        [0.10089, -0.10746, 0.34917],
    ],
    dtype=np.float64,
)
_CANONICAL_MAP_INDICES = (
    30,
    31,
    35,
    52,
    62,
    63,
    55,
    66,
    65,
    1,
    9,
    8,
    7,
    15,
    40,
    42,
    27,
    39,
    37,
)


class FaceEngine:
    """Runs the five Buffalo_L graphs through Triton."""

    def __init__(self, client: TritonClientProtocol) -> None:
        self._client = client
        self._det_threshold = float(config.get("face_engine.det_threshold", 0.6))
        det_size = config.get("face_engine.det_size", [640, 640])
        if det_size != [640, 640]:
            raise ValueError("The Triton SCRFD contract requires face_engine.det_size [640, 640]")

        self.model_profile = str(config.get("face_engine.model_profile", "full"))
        if self.model_profile not in {"full", "int8"}:
            raise ValueError("face_engine.model_profile must be 'full' or 'int8'")

        self.endpoint = str(config.get("face_engine.triton_url", "triton:8701")).strip()
        if not self.endpoint:
            raise ValueError("face_engine.triton_url must not be empty")

        self.model_names = {
            "detector": str(config.get("face_engine.models.detector", "face-detector-scrfd")),
            "recognition": str(
                config.get("face_engine.models.recognition", "face-recognition-arcface")
            ),
            "landmark_2d": str(config.get("face_engine.models.landmark_2d", "face-landmark-2d106")),
            "landmark_3d": str(config.get("face_engine.models.landmark_3d", "face-landmark-3d68")),
            "genderage": str(
                config.get("face_engine.models.genderage", "face-attribute-genderage")
            ),
        }
        if any(not name.strip() for name in self.model_names.values()):
            raise ValueError("All face_engine.models entries must be non-empty")
        if len(set(self.model_names.values())) != len(self.model_names):
            raise ValueError("Each face_engine.models entry must name a distinct Triton model")
        logger.info(
            "FaceEngine ready: backend=triton endpoint=%s profile=%s",
            self.endpoint,
            self.model_profile,
        )

    async def validate_models(self) -> None:
        """Fail startup unless every configured Buffalo_L model is ready."""
        readiness = await asyncio.gather(
            *(self._client.is_model_ready(name) for name in self.model_names.values())
        )
        missing = [
            name
            for name, ready in zip(self.model_names.values(), readiness, strict=True)
            if not ready
        ]
        if missing:
            raise RuntimeError(f"Required Triton face models are not ready: {', '.join(missing)}")

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces and run all Buffalo_L enrichment graphs."""
        detector_input, det_scale = _prepare_scrfd(image)
        outputs = cast(
            dict[str, FloatArray],
            await self._client.infer(
                self.model_names["detector"],
                [("input.1", detector_input)],
                _SCRFD_OUTPUTS,
            ),
        )
        detections, keypoints = _decode_scrfd(
            outputs,
            det_scale=det_scale,
            threshold=self._det_threshold,
        )
        if len(detections) == 0:
            return []

        return list(
            await asyncio.gather(
                *(
                    self._enrich_face(image, detection, keypoint)
                    for detection, keypoint in zip(detections, keypoints, strict=True)
                )
            )
        )

    async def _enrich_face(
        self,
        image: np.ndarray,
        detection: np.ndarray,
        keypoints: np.ndarray,
    ) -> DetectedFace:
        bbox = detection[:4]
        arcface_input = _prepare_arcface(image, keypoints)
        landmark_input, landmark_matrix = _prepare_bbox_crop(image, bbox, 192)
        genderage_input, _ = _prepare_bbox_crop(image, bbox, 96)

        recognition, landmark_2d, landmark_3d, genderage = await asyncio.gather(
            self._client.infer(
                self.model_names["recognition"],
                [("input.1", arcface_input)],
                ["683"],
            ),
            self._client.infer(
                self.model_names["landmark_2d"],
                [("data", landmark_input)],
                ["fc1"],
            ),
            self._client.infer(
                self.model_names["landmark_3d"],
                [("data", landmark_input)],
                ["fc1"],
            ),
            self._client.infer(
                self.model_names["genderage"],
                [("data", genderage_input)],
                ["fc1"],
            ),
        )

        embedding = np.ravel(recognition["683"]).astype(np.float32, copy=True)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0:
            raise RuntimeError("ArcFace returned a zero-length embedding")
        embedding = (embedding / norm).astype(np.float32, copy=False)

        landmarks_2d = _decode_landmarks(landmark_2d["fc1"], 106, 2, landmark_matrix)
        landmarks_3d = _decode_landmarks(landmark_3d["fc1"], 68, 3, landmark_matrix)
        yaw_deg, pitch_deg, roll_deg = _estimate_head_pose_from_landmarks(landmarks_3d)

        attributes = np.asarray(genderage["fc1"], dtype=np.float32).reshape(-1)
        gender = int(np.argmax(attributes[:2])) if attributes.size >= 2 else None
        age = int(np.rint(attributes[2] * 100.0)) if attributes.size >= 3 else None

        return DetectedFace(
            bbox=[float(value) for value in bbox],
            embedding=embedding,
            det_score=float(detection[4]),
            landmarks=landmarks_2d,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            gender=gender,
            age=age,
        )

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between normalized embeddings."""
        return float(np.dot(emb1, emb2))


def _prepare_scrfd(image: np.ndarray) -> tuple[np.ndarray, float]:
    input_width = input_height = 640
    image_ratio = float(image.shape[0]) / image.shape[1]
    if image_ratio > 1.0:
        resized_height = input_height
        resized_width = int(resized_height / image_ratio)
    else:
        resized_width = input_width
        resized_height = int(resized_width * image_ratio)
    scale = float(resized_height) / image.shape[0]
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = ((rgb.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), scale


def _decode_scrfd(
    outputs: dict[str, FloatArray],
    *,
    det_scale: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = [np.asarray(outputs[name]) for name in _SCRFD_OUTPUTS]
    scores_list: list[np.ndarray] = []
    boxes_list: list[np.ndarray] = []
    keypoints_list: list[np.ndarray] = []
    for index, stride in enumerate((8, 16, 32)):
        scores = values[index].reshape(-1, 1)
        box_predictions = values[index + 3].reshape(-1, 4) * stride
        keypoint_predictions = values[index + 6].reshape(-1, 10) * stride
        height = 640 // stride
        width = 640 // stride
        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        centers = np.stack((x_coordinates, y_coordinates), axis=-1)
        centers = (centers * stride).reshape(-1, 2)
        centers = np.stack([centers] * 2, axis=1).reshape(-1, 2)
        active = np.where(scores[:, 0] >= threshold)[0]
        scores_list.append(scores[active])
        boxes_list.append(_distance_to_boxes(centers, box_predictions)[active])
        decoded_keypoints = _distance_to_keypoints(centers, keypoint_predictions)
        keypoints_list.append(decoded_keypoints[active].reshape(-1, 5, 2))

    if not any(len(scores) for scores in scores_list):
        return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

    scores = np.vstack(scores_list)
    boxes = np.vstack(boxes_list) / det_scale
    keypoints = np.vstack(keypoints_list) / det_scale
    order = scores[:, 0].argsort()[::-1]
    detections = np.hstack((boxes, scores)).astype(np.float32, copy=False)[order]
    keypoints = keypoints[order]
    keep = _nms(detections, 0.4)
    return detections[keep], keypoints[keep]


def _distance_to_boxes(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            points[:, 0] - distances[:, 0],
            points[:, 1] - distances[:, 1],
            points[:, 0] + distances[:, 2],
            points[:, 1] + distances[:, 3],
        ],
        axis=-1,
    )


def _distance_to_keypoints(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    coordinates = []
    for index in range(0, distances.shape[1], 2):
        coordinates.append(points[:, 0] + distances[:, index])
        coordinates.append(points[:, 1] + distances[:, index + 1])
    return np.stack(coordinates, axis=-1)


def _nms(detections: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2, scores = detections.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = width * height / (areas[current] + areas[order[1:]] - width * height)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _prepare_arcface(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    matrix = _similarity_transform(keypoints.astype(np.float32), _ARCFACE_DST)
    aligned = cv2.warpAffine(image, matrix, (112, 112), borderValue=0.0)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    tensor = ((rgb.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor)


def _prepare_bbox_crop(
    image: np.ndarray,
    bbox: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    center_x = float(bbox[0] + bbox[2]) / 2.0
    center_y = float(bbox[1] + bbox[3]) / 2.0
    scale = output_size / (max(width, height) * 1.5)
    matrix = np.array(
        [
            [scale, 0.0, output_size / 2.0 - center_x * scale],
            [0.0, scale, output_size / 2.0 - center_y * scale],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(image, matrix, (output_size, output_size), borderValue=0.0)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), matrix


def _similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    diagonal = np.ones(2, dtype=np.float32)
    if np.linalg.det(covariance) < 0:
        diagonal[-1] = -1
    rotation = left @ np.diag(diagonal) @ right_t
    variance = np.sum(source_centered**2) / source.shape[0]
    scale = float(np.sum(singular_values * diagonal) / variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    return np.hstack((scale * rotation, translation[:, None])).astype(np.float32)


def _decode_landmarks(
    output: np.ndarray,
    count: int,
    dimensions: int,
    transform: np.ndarray,
) -> np.ndarray:
    flattened = np.asarray(output, dtype=np.float32).reshape(-1)
    prediction = np.array(
        flattened[-count * dimensions :],
        dtype=np.float32,
        copy=True,
    ).reshape(count, dimensions)
    prediction[:, :2] = (prediction[:, :2] + 1.0) * 96.0
    if dimensions == 3:
        prediction[:, 2] *= 96.0
    inverse = cv2.invertAffineTransform(transform)
    xy = np.hstack((prediction[:, :2], np.ones((count, 1), dtype=np.float32))) @ inverse.T
    if dimensions == 2:
        return xy.astype(np.float32)
    scale = float(np.sqrt(inverse[0, 0] ** 2 + inverse[0, 1] ** 2))
    return np.column_stack((xy, prediction[:, 2] * scale)).astype(np.float32)


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decode a base64 string with or without a data URI prefix."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_str)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image from base64 data")
    return image


def _estimate_head_pose(face: LandmarkFaceProtocol) -> tuple[float, float, float]:
    """Compatibility wrapper used by existing tests."""
    return _estimate_head_pose_from_landmarks(face.landmark_3d_68)


def _estimate_head_pose_from_landmarks(
    landmarks: np.ndarray | None,
) -> tuple[float, float, float]:
    if landmarks is None or landmarks.shape[0] < 68:
        raise ValueError("The 3D landmark model did not return 68 landmarks")
    image_points = np.array(
        [
            [float(landmarks[index, 0]), float(landmarks[index, 1])]
            for index in _CANONICAL_MAP_INDICES
        ],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[640.0, 0.0, 320.0], [0.0, 640.0, 320.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rotation_vector, _ = cv2.solvePnP(
        _CANONICAL_FACE_3D,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Head-pose solvePnP failed")
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    else:
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
        roll = 0.0
    return (
        float(np.degrees(yaw)),
        float(np.degrees(pitch)),
        float(np.degrees(roll)),
    )
