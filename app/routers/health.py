"""Health check and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest
from starlette.responses import Response

from app import config

router = APIRouter()

# Bounded-label Prometheus gauge for calibration health state.
# Four possible states; current state is set to 1.0, others to 0.0.
_CALIBRATION_HEALTH = Gauge(
    "pid_calibration_health_state",
    "1 if this calibration health state is currently active, 0 otherwise",
    ["state"],
)
for _s in ("ready", "degraded_missing", "degraded_incompatible", "degraded_invalid"):
    _CALIBRATION_HEALTH.labels(state=_s).set(0.0)


@router.get("/health")
async def health(request: Request):
    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store
    evaluator = request.app.state.calibration_evaluator

    cal_state = evaluator.health()
    for _s in ("ready", "degraded_missing", "degraded_incompatible", "degraded_invalid"):
        _CALIBRATION_HEALTH.labels(state=_s).set(1.0 if _s == cal_state else 0.0)

    return {
        "status": "ok",
        "inference_backend": "triton",
        "triton_endpoint": engine.endpoint,
        "model_profile": engine.model_profile,
        "enrolled_members": await store.member_count(),
        "model": config.get("face_engine.model_name", "buffalo_l"),
        "models": engine.model_names,
        "calibration": {
            "status": cal_state,
            "artifact_version": evaluator.artifact_version(),
            "arcface_model_version": evaluator.arcface_model_version(),
            "preprocessing_version": evaluator.preprocessing_version(),
        },
    }


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    """Expose the default prometheus_client registry in the Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
