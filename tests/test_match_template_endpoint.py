"""Wire contract of POST /api/v1/match-template and of ``return_embedding`` on /identify.

The routers, the template match service, the in-memory gallery repository, the aggregator, and the
calibration evaluator run for real; only face detection is a fake engine.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import TEMPLATE_PROFILE, CalibrationEvaluator
from app.routers import identification, templates
from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality
from app.services.motion_detector import MotionDetector
from app.services.template_aggregation import (
    EMBEDDING_DIM,
    QualityWeightedMeanAggregator,
    decode_embedding_f16,
    encode_embedding_f16,
)
from app.services.template_match import TemplateMatchService
from tests.calibration_fakes import disabled_capture
from tests.gallery import DEFAULT_THRESHOLDS, one_member_store

_QUALITY_VERSION = "heuristic-v1"


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


_ALPHA = _unit(1)
_BETA = _unit(2)


def _noisy_alpha(seed: int) -> np.ndarray:
    """A face of alpha at cosine about 0.7: one weak glimpse."""
    v = _ALPHA + _unit(100 + seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _RecordingVisitorStore:
    def __init__(self) -> None:
        self.calls = 0

    async def record_sighting(self, *args: object, **kwargs: object) -> None:
        self.calls += 1


def _gallery(*members: tuple[str, np.ndarray]) -> InMemoryEnrollmentRepository:
    repository = InMemoryEnrollmentRepository()

    async def _seed() -> None:
        for person_id, embedding in members:
            await repository.upsert_member(person_id, person_id.title())
            await repository.insert_embeddings(person_id, [embedding])
            await repository.recompute_centroid(person_id)

    asyncio.run(_seed())
    return repository


def _client(
    template_evaluator: CalibrationEvaluator | None = None,
    repository: InMemoryEnrollmentRepository | None = None,
) -> tuple[TestClient, _RecordingVisitorStore]:
    app = FastAPI()
    app.include_router(templates.router)
    app.state.template_matcher = TemplateMatchService(
        repository or _gallery(("alpha", _ALPHA), ("beta", _BETA)),
        QualityWeightedMeanAggregator(),
        DEFAULT_THRESHOLDS,
    )
    app.state.template_calibration_evaluator = template_evaluator or CalibrationEvaluator(
        None, "degraded_missing"
    )
    app.state.face_quality_version = _QUALITY_VERSION
    visitors = _RecordingVisitorStore()
    app.state.visitor_store = visitors
    return TestClient(app), visitors


def _member(embedding: np.ndarray, score: float = 0.5, **quality: Any) -> dict[str, Any]:
    return {
        "embedding_f16_b64": encode_embedding_f16(embedding),
        "quality": {
            "score": score,
            "resolution": 0.4,
            "sharpness": 0.6,
            "detection": 0.9,
            "pitch": 1.0,
            "version": _QUALITY_VERSION,
        }
        | quality,
    }


def _body(members: list[dict[str, Any]], version: str = "buffalo_l") -> dict[str, Any]:
    return {"members": members, "embedding_model_version": version}


def test_a_template_of_weak_faces_matches_better_than_any_member() -> None:
    client, visitors = _client()
    faces = [_noisy_alpha(i) for i in range(6)]

    response = client.post(
        "/api/v1/match-template",
        json=_body([_member(f, score=0.3 + 0.1 * i) for i, f in enumerate(faces)]),
    )

    assert response.status_code == 200
    body = response.json()
    best_member = max(float(f @ _ALPHA) for f in faces)
    assert body["person_id"] == "alpha"
    assert body["recognition_state"] == "recognized"
    assert body["best_candidate_id"] == "alpha"
    assert body["raw_similarity"] > best_member + 0.1
    assert body["second_best_similarity"] < body["raw_similarity"]
    assert body["member_count"] == 6
    assert body["aggregator_version"] == "qwmean-v1"
    assert body["quality_version"] == _QUALITY_VERSION
    # Three best members by score: 0.8, 0.7, 0.6.
    assert body["template_quality"]["score"] == pytest.approx(0.7, abs=1e-6)
    assert body["calibrated_confidence"] is None
    assert (body["calibration_profile"], body["calibration_status"]) == ("", "degraded_missing")
    # Embeddings in, no image: nothing reaches visitor clustering.
    assert visitors.calls == 0


def test_a_one_member_gallery_has_no_second_best() -> None:
    client, _ = _client(repository=_gallery(("alpha", _ALPHA)))
    response = client.post(
        "/api/v1/match-template", json=_body([_member(_noisy_alpha(1)), _member(_noisy_alpha(2))])
    )
    assert response.status_code == 200
    assert response.json()["second_best_similarity"] is None


@pytest.mark.parametrize("count", [1, 17])
def test_member_count_outside_two_to_sixteen_is_rejected(count: int) -> None:
    client, _ = _client()
    response = client.post(
        "/api/v1/match-template", json=_body([_member(_noisy_alpha(i)) for i in range(count)])
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (
            _body([_member(_ALPHA), _member(_BETA)], version="antelopev2"),
            "template.embedding_model_version_mismatch",
        ),
        (
            _body([_member(_ALPHA), _member(_BETA, version="heuristic-v2")]),
            "template.quality_version_mismatch",
        ),
        (
            _body([_member(_ALPHA), {**_member(_BETA), "embedding_f16_b64": "AAAA"}]),
            "template.embedding_invalid",
        ),
        (
            _body([_member(_ALPHA, score=0.0), _member(_BETA, score=0.0)]),
            "template.aggregation_invalid",
        ),
    ],
)
def test_contract_violations_are_422_with_a_code(body: dict[str, Any], code: str) -> None:
    client, _ = _client()
    response = client.post("/api/v1/match-template", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == code


_TEMPLATE_ARTIFACT: dict[str, Any] = {
    "schema_version": 2,
    "artifact_version": "t-v1",
    "method": "logistic_v2",
    "arcface_model_version": "buffalo_l",
    "model_profile": "full",
    "preprocessing_version": "v1",
    "fitted_at": "2026-11-11T09:00:00+00:00",
    "profile": "template",
    "aggregation": "qwmean-v1",
    "quality_version": _QUALITY_VERSION,
    "features": ["similarity", "quality_score"],
    "coefficients": [10.0, 2.0],
    "intercept": -6.0,
    "covered_identities": ["alpha"],
    "fit_basis": "household_only",
}


def _evaluator(tmp_path: Path, artifact: dict[str, Any]) -> CalibrationEvaluator:
    path = tmp_path / "template.json"
    path.write_text(json.dumps(artifact))
    return CalibrationEvaluator.from_artifact_path(
        path,
        expected_arcface_model_version="buffalo_l",
        expected_model_profile="full",
        expected_preprocessing_version="v1",
        expected_quality_version=_QUALITY_VERSION,
        expected_calibration_profile=TEMPLATE_PROFILE,
        expected_aggregation="qwmean-v1",
    )


def test_a_template_artifact_calibrates_the_template_score(tmp_path: Path) -> None:
    client, _ = _client(_evaluator(tmp_path, _TEMPLATE_ARTIFACT))

    body = client.post(
        "/api/v1/match-template",
        json=_body([_member(_noisy_alpha(i), score=0.5) for i in range(4)]),
    ).json()

    z = -6.0 + 10.0 * body["raw_similarity"] + 2.0 * body["template_quality"]["score"]
    assert body["calibrated_confidence"] == pytest.approx(1.0 / (1.0 + math.exp(-z)), abs=1e-6)
    assert (body["calibration_profile"], body["calibration_status"]) == ("template", "ready")
    assert body["calibration_artifact_version"] == "t-v1"


@pytest.mark.parametrize(
    "changes",
    [{"profile": "single_frame", "aggregation": None}, {"aggregation": "caface-v1"}],
)
def test_a_single_frame_or_other_aggregator_artifact_never_calibrates_templates(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    evaluator = _evaluator(tmp_path, _TEMPLATE_ARTIFACT | changes)
    client, _ = _client(evaluator)

    body = client.post(
        "/api/v1/match-template",
        json=_body([_member(_noisy_alpha(i)) for i in range(4)]),
    ).json()

    assert evaluator.health() == "degraded_incompatible"
    assert body["calibrated_confidence"] is None
    assert body["calibration_status"] == "degraded_incompatible"


# --- /identify return_embedding ----------------------------------------------------------------


class _OneFaceEngine:
    model_profile = "full"

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        face = DetectedFace(bbox=[10.0, 20.0, 60.0, 80.0], embedding=_ALPHA, det_score=0.8)
        face.quality = FaceQuality(
            score=0.5,
            resolution=0.4,
            sharpness=0.6,
            detection=0.9,
            pitch=1.0,
            version=_QUALITY_VERSION,
        )
        return [face]


def _identify_client() -> TestClient:
    app = FastAPI()
    app.include_router(identification.router)
    app.state.face_engine = _OneFaceEngine()
    app.state.enrollment_store = one_member_store(
        _OneFaceEngine(), person_id="alpha", name="Alpha", probe=_ALPHA, similarity=0.55
    )
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.visitor_store = _RecordingVisitorStore()
    app.state.calibration_capture = disabled_capture()
    app.state.motion_detector = MotionDetector()
    return TestClient(app)


def _image() -> str:
    ok, buffer = cv2.imencode(".png", np.full((96, 96, 3), 127, dtype=np.uint8))
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def test_identify_returns_the_embedding_only_when_asked() -> None:
    client = _identify_client()

    plain = client.post("/api/v1/identify", json={"image": _image()}).json()["faces"][0]
    asked = client.post(
        "/api/v1/identify", json={"image": _image(), "return_embedding": True}
    ).json()["faces"][0]

    assert plain["embedding_f16_b64"] is None
    assert plain["embedding_model_version"] is None
    np.testing.assert_allclose(decode_embedding_f16(asked["embedding_f16_b64"]), _ALPHA, atol=1e-3)
    assert asked["embedding_model_version"] == "buffalo_l"


def test_identify_batch_never_carries_embeddings() -> None:
    client = _identify_client()
    frames = client.post(
        "/api/v1/identify-batch", json={"images": [_image()], "include_motion": False}
    ).json()["frames"]
    assert frames[0]["faces"][0]["embedding_f16_b64"] is None
