"""/health reports the single-frame and the template calibration separately."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import CalibrationEvaluator
from app.routers import health


class _Engine:
    endpoint = "triton:8701"
    model_profile = "full"
    model_names: dict[str, str] = {}


class _Store:
    async def member_count(self) -> int:
        return 2


def test_health_has_a_template_calibration_block() -> None:
    app = FastAPI()
    app.include_router(health.router)
    app.state.face_engine = _Engine()
    app.state.enrollment_store = _Store()
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.template_calibration_evaluator = CalibrationEvaluator(None, "degraded_incompatible")

    body = TestClient(app).get("/health").json()

    assert body["calibration"]["status"] == "degraded_missing"
    assert body["calibration"]["gallery_representation"] == ""
    assert body["gallery_representation"] == "centroid"
    assert body["active_gallery_representation"] == "centroid"
    assert body["template_calibration"] == {
        "status": "degraded_incompatible",
        "artifact_version": None,
        "recognizer_version": "",
        "profile": "",
        "aggregation": None,
        "quality_version": "",
        "gallery_representation": "",
        "covered_identities": None,
    }
