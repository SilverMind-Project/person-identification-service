"""Wire contract for face quality on /identify and /identify-batch, and its startup validation.

The router, the enrollment store's identify_all carry-forward, and the response models run for
real; detection is a fake engine that scores its face with the real heuristic scorer.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import main
from app.calibration.evaluator import CalibrationEvaluator
from app.models.identification import FaceDetection
from app.routers import identification
from app.services.face_models import DetectedFace
from app.services.face_quality import (
    FaceQualityInputs,
    HeuristicFaceQualityScorer,
    HeuristicQualityParams,
)
from app.services.motion_detector import MotionDetector
from tests.calibration_fakes import disabled_capture
from tests.gallery import one_member_store

# Eyes 23 px apart: resolution component 0.5 under the default ramp (10 to 36 px).
_KEYPOINTS = np.array(
    [[20.0, 40.0], [43.0, 40.0], [31.0, 55.0], [24.0, 66.0], [40.0, 66.0]], dtype=np.float32
)
_SCORER = HeuristicFaceQualityScorer(HeuristicQualityParams())


class _ScoringEngine:
    model_profile = "full"

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0] = 1.0
        face = DetectedFace(
            bbox=[10.0, 20.0, 60.0, 80.0],
            embedding=embedding,
            det_score=0.7,
            keypoints_5_px=_KEYPOINTS,
            pitch_deg=30.0,
        )
        face.quality = _SCORER.score(
            FaceQualityInputs(
                interocular_px=face.geometry.interocular_px,
                det_score=face.det_score,
                pitch_deg=face.pitch_deg,
                aligned_gray_112=np.zeros((112, 112), dtype=np.uint8),
            )
        )
        return [face]


class _NoopVisitorStore:
    async def record_sighting(self, *args: object, **kwargs: object) -> None:
        return None


def _encoded_image() -> str:
    ok, buffer = cv2.imencode(".png", np.full((96, 96, 3), 127, dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(identification.router)
    app.state.face_engine = _ScoringEngine()
    app.state.enrollment_store = one_member_store(
        _ScoringEngine(),
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


_EXPECTED_QUALITY = {
    "resolution": pytest.approx(0.5),
    "sharpness": 0.0,  # a flat aligned crop has no edges
    "detection": pytest.approx(0.5),
    "pitch": pytest.approx(0.5),
    "version": "heuristic-v1",
    "score": pytest.approx(0.5**0.45 * 0.01**0.25 * 0.5**0.15 * 0.5**0.15),
}


def test_identify_serializes_quality_with_every_component_and_version(client: TestClient) -> None:
    [face] = client.post("/api/v1/identify", json={"image": _encoded_image()}).json()["faces"]
    assert face["quality"] == _EXPECTED_QUALITY


def test_identify_batch_serializes_quality(client: TestClient) -> None:
    response = client.post(
        "/api/v1/identify-batch", json={"images": [_encoded_image()], "include_motion": False}
    )
    [frame] = response.json()["frames"]
    assert frame["faces"][0]["quality"] == _EXPECTED_QUALITY


def test_quality_defaults_to_null_for_an_unscored_face() -> None:
    dumped = FaceDetection(person_id="p", name="n", similarity=0.5, bbox=[0, 0, 1, 1]).model_dump()
    assert dumped["quality"] is None


async def test_startup_fails_on_an_unknown_scorer(monkeypatch: pytest.MonkeyPatch) -> None:
    real_get = main.config.get

    def _config_get(key: str, default: object | None = None) -> object | None:
        if key == "face_quality":
            return {"scorer": "not-a-scorer"}
        if key == "database.dsn":
            # Never let this lifespan reach a real database, whatever the environment holds.
            return None
        return real_get(key, default)

    monkeypatch.setattr(main.config, "get", _config_get)
    with pytest.raises(ValueError, match=r"face_quality\.scorer"):
        async with main.lifespan(FastAPI()):
            pass
