"""Wire contract for face geometry on /identify and /identify-batch.

The router, the enrollment store's identify_all carry-forward, and the response models run for
real; only the database lookup, Triton inference, and visitor persistence are replaced.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import CalibrationEvaluator
from app.models.identification import FaceDetection
from app.routers import identification
from app.services.face_models import DetectedFace
from app.services.motion_detector import MotionDetector
from tests.calibration_fakes import disabled_capture
from tests.gallery import one_member_store

_IMAGE_WIDTH_PX = 96
_IMAGE_HEIGHT_PX = 128
# Eyes 30 px apart horizontally and 40 px vertically: 50 px interocular distance.
_KEYPOINTS = np.array(
    [[20.0, 40.0], [50.0, 80.0], [35.0, 90.0], [25.0, 100.0], [45.0, 100.0]], dtype=np.float32
)
_BBOX = [10.0, 20.0, 70.0, 110.0]


class _FakeEngine:
    model_profile = "full"

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        return [
            DetectedFace(
                bbox=list(_BBOX), embedding=embedding, det_score=0.9, keypoints_5_px=_KEYPOINTS
            )
        ]


class _NoopVisitorStore:
    async def record_sighting(self, *args: object, **kwargs: object) -> None:
        return None


def _encoded_image() -> str:
    image = np.full((_IMAGE_HEIGHT_PX, _IMAGE_WIDTH_PX, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(identification.router)
    app.state.face_engine = _FakeEngine()
    app.state.enrollment_store = one_member_store(
        _FakeEngine(),
        person_id="resident-alpha",
        name="Resident Alpha",
        probe=np.eye(512, dtype=np.float32)[0],
        similarity=0.62,
    )
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.visitor_store = _NoopVisitorStore()
    app.state.calibration_capture = disabled_capture()
    app.state.motion_detector = MotionDetector()
    return TestClient(app)


def test_face_detection_geometry_defaults() -> None:
    dumped = FaceDetection(person_id="p", name="n", similarity=0.5, bbox=[0, 0, 1, 1]).model_dump()
    assert dumped["face_width_px"] == 0.0
    assert dumped["face_height_px"] == 0.0
    assert dumped["interocular_px"] is None


def test_identify_returns_face_geometry_and_image_size(client: TestClient) -> None:
    response = client.post("/api/v1/identify", json={"image": _encoded_image()})
    assert response.status_code == 200
    body = response.json()
    assert body["image_width_px"] == _IMAGE_WIDTH_PX
    assert body["image_height_px"] == _IMAGE_HEIGHT_PX
    [face] = body["faces"]
    assert face["face_width_px"] == pytest.approx(60.0)
    assert face["face_height_px"] == pytest.approx(90.0)
    assert face["interocular_px"] == pytest.approx(50.0)


def test_identify_batch_frames_carry_geometry_and_image_size(client: TestClient) -> None:
    response = client.post(
        "/api/v1/identify-batch",
        json={"images": [_encoded_image(), "not-an-image"], "include_motion": False},
    )
    assert response.status_code == 200
    decoded, undecodable = response.json()["frames"]
    assert (decoded["image_width_px"], decoded["image_height_px"]) == (
        _IMAGE_WIDTH_PX,
        _IMAGE_HEIGHT_PX,
    )
    assert decoded["faces"][0]["interocular_px"] == pytest.approx(50.0)
    assert (undecodable["image_width_px"], undecodable["image_height_px"]) == (0, 0)
    assert undecodable["faces"] == []


def test_identify_face_json_has_no_confidence_key(client: TestClient) -> None:
    # A face carries its match score as similarity (and the raw_similarity alias); the old
    # "confidence" key, a mislabelled raw similarity, is gone from the wire contract.
    [face] = client.post("/api/v1/identify", json={"image": _encoded_image()}).json()["faces"]
    assert "confidence" not in face
    assert face["raw_similarity"] == face["similarity"] == pytest.approx(0.62)
