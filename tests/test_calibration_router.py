"""Calibration review endpoints and the /identify capture hook, over in-memory peers."""

from __future__ import annotations

import asyncio
import base64
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import CalibrationEvaluator
from app.routers import calibration, identification
from app.services.calibration_candidates import (
    CalibrationCandidate,
    InMemoryCalibrationCandidateRepository,
)
from app.services.calibration_capture import CalibrationCaptureService, CapturePolicy
from app.services.face_models import DetectedFace
from tests.calibration_fakes import VERSIONS, InMemoryCropStore
from tests.gallery import one_member_store

_T = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def _candidate(index: int, *, minutes: float, camera: str = "cam-1") -> CalibrationCandidate:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[index % 512] = 1.0
    return CalibrationCandidate(
        candidate_id=str(uuid.UUID(int=index + 1)),
        captured_at=_T + timedelta(minutes=minutes),
        camera_id=camera,
        source="cts_identify",
        crop_object_key=f"calibration/{index}.png",
        embedding=embedding,
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        quality_version="heuristic-v1",
        quality_score=0.35,
        quality_components={"resolution": 0.3},
        capture_profile={"crop_jpeg_quality": 95},
        interocular_px=18.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        nearest_person_id="resident-alpha",
        nearest_similarity=0.5,
    )


class _NoFaces:
    model_profile = "full"

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return []


class _SpyCapture:
    enabled = True

    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def offer(self, image: Any, faces: Any, identities: Any, context: Any) -> int:
        self.contexts.append(context)
        return 0


@pytest.fixture
def repo() -> InMemoryCalibrationCandidateRepository:
    repo = InMemoryCalibrationCandidateRepository()
    # Two walks past cam-1 (gap of 10 minutes) and one past cam-2.
    for index, (minutes, camera) in enumerate(
        [(0, "cam-1"), (1, "cam-1"), (12, "cam-1"), (13, "cam-2"), (30, "cam-1")]
    ):
        asyncio.run(repo.insert(_candidate(index, minutes=minutes, camera=camera)))
    return repo


@pytest.fixture
def crops() -> InMemoryCropStore:
    store = InMemoryCropStore()
    store.objects = {f"calibration/{i}.png": b"png" for i in range(5)}
    return store


@pytest.fixture
def client(repo: InMemoryCalibrationCandidateRepository, crops: InMemoryCropStore) -> TestClient:
    app = FastAPI()
    app.include_router(calibration.router)
    app.state.calibration_candidates = repo
    app.state.calibration_capture = CalibrationCaptureService(
        repo, crops, CapturePolicy(), VERSIONS
    )
    app.state.enrollment_store = one_member_store(
        _NoFaces(),
        person_id="resident-alpha",
        name="Resident Alpha",
        probe=np.eye(512, dtype=np.float32)[0],
        similarity=0.5,
    )
    return TestClient(app)


def _label(client: TestClient, index: int, **body: Any) -> Any:
    payload = {"labeled_by": "operator"} | body
    return client.post(
        f"/api/v1/calibration/candidates/{uuid.UUID(int=index + 1)}/label", json=payload
    )


def test_list_pages_newest_first_with_a_cursor(client: TestClient) -> None:
    first = client.get("/api/v1/calibration/candidates", params={"limit": 3}).json()
    second = client.get(
        "/api/v1/calibration/candidates", params={"limit": 3, "cursor": first["next_cursor"]}
    ).json()
    order = [c["candidate_id"] for c in first["candidates"] + second["candidates"]]
    assert order == [str(uuid.UUID(int=i + 1)) for i in (4, 3, 2, 1, 0)]
    assert second["next_cursor"] is None
    assert "embedding" not in first["candidates"][0]
    assert first["candidates"][0]["crop_object_key"] == "calibration/4.png"


def test_a_bad_cursor_is_a_422(client: TestClient) -> None:
    response = client.get("/api/v1/calibration/candidates", params={"cursor": "nonsense"})
    assert response.status_code == 422


def test_label_rules(client: TestClient) -> None:
    ok = _label(client, 0, label_kind="member", label_identity_id="resident-alpha")
    assert ok.status_code == 200
    assert (ok.json()["label_state"], ok.json()["label_identity_id"]) == (
        "labeled",
        "resident-alpha",
    )
    assert _label(client, 1, label_kind="member", label_identity_id="nobody").status_code == 422
    assert _label(client, 1, label_kind="member").status_code == 422
    assert (
        _label(client, 1, label_kind="not_enrolled", label_identity_id="resident-alpha").status_code
        == 422
    )
    assert _label(client, 99, label_kind="not_a_face").status_code == 404
    unlabeled = client.get("/api/v1/calibration/candidates", params={"state": "unlabeled"}).json()
    assert len(unlabeled["candidates"]) == 4


def test_reject_and_delete(client: TestClient, crops: InMemoryCropStore) -> None:
    rejected = client.post(f"/api/v1/calibration/candidates/{uuid.UUID(int=2)}/reject")
    assert rejected.json()["label_state"] == "rejected"
    assert (
        client.post(f"/api/v1/calibration/candidates/{uuid.UUID(int=99)}/reject").status_code == 404
    )
    assert client.delete(f"/api/v1/calibration/candidates/{uuid.UUID(int=1)}").status_code == 204
    assert "calibration/0.png" not in crops.objects
    assert client.delete(f"/api/v1/calibration/candidates/{uuid.UUID(int=1)}").status_code == 404


def test_summary_counts_episodes_per_identity(client: TestClient) -> None:
    for index in (0, 1, 2, 3):
        _label(client, index, label_kind="member", label_identity_id="resident-alpha")
    _label(client, 4, label_kind="not_enrolled")

    summary = client.get("/api/v1/calibration/summary").json()
    assert summary["by_state"] == {"unlabeled": 0, "labeled": 5, "rejected": 0}
    assert summary["by_label_kind"] == {"member": 4, "not_enrolled": 1}
    assert summary["by_camera"] == {"cam-1": 4, "cam-2": 1}
    assert summary["by_quality_decile"] == {"3": 5}
    # cam-1 at 0 and 1 minutes is one walk, 12 minutes starts another; cam-2 is a third.
    assert summary["identities"] == [
        {
            "identity_id": "resident-alpha",
            "labeled_faces": 4,
            "episodes": 3,
            "target_labeled_faces": 150,
        }
    ]
    assert summary["not_enrolled_episodes"] == 1
    assert summary["capture_enabled"] is False


def _identify_client(capture: _SpyCapture) -> TestClient:
    class _NoopVisitorStore:
        async def record_sighting(self, *args: object, **kwargs: object) -> None:
            return None

    app = FastAPI()
    app.include_router(identification.router)
    app.state.face_engine = _NoFaces()
    app.state.enrollment_store = one_member_store(
        _NoFaces(),
        person_id="resident-alpha",
        name="A",
        probe=np.eye(512, dtype=np.float32)[0],
        similarity=0.5,
    )
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.visitor_store = _NoopVisitorStore()
    app.state.calibration_capture = capture
    return TestClient(app)


def _image() -> str:
    ok, buffer = cv2.imencode(".png", np.full((32, 32, 3), 127, dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def test_identify_hands_its_capture_context_to_capture() -> None:
    capture = _SpyCapture()
    client = _identify_client(capture)
    context = {
        "camera_id": "cam-7",
        "captured_at": "2026-10-21T09:00:00+00:00",
        "crop_pad_top_frac": 0.15,
        "crop_pad_side_frac": 0.1,
        "crop_jpeg_quality": 95,
    }
    assert (
        client.post(
            "/api/v1/identify", json={"image": _image(), "capture_context": context}
        ).status_code
        == 200
    )
    assert client.post("/api/v1/identify", json={"image": _image()}).status_code == 200

    sent, absent = capture.contexts
    assert (sent.camera_id, sent.captured_at, sent.crop_jpeg_quality) == ("cam-7", _T, 95)
    assert absent is None


def test_naive_capture_time_is_rejected() -> None:
    client = _identify_client(_SpyCapture())
    context = {
        "camera_id": "cam-7",
        "captured_at": "2026-10-21T09:00:00",
        "crop_pad_top_frac": 0.15,
        "crop_pad_side_frac": 0.1,
        "crop_jpeg_quality": 95,
    }
    response = client.post("/api/v1/identify", json={"image": _image(), "capture_context": context})
    assert response.status_code == 422
