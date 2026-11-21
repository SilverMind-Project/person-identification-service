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
    template_evaluator = request.app.state.template_calibration_evaluator

    cal_state = evaluator.health()
    for _s in ("ready", "degraded_missing", "degraded_incompatible", "degraded_invalid"):
        _CALIBRATION_HEALTH.labels(state=_s).set(1.0 if _s == cal_state else 0.0)

    active_profile = getattr(engine, "profile", None)
    active_profile_name = (
        active_profile.name
        if active_profile is not None
        else str(config.get("face_engine.active_profile", "buffalo_l"))
    )
    recognizer_version = str(
        getattr(
            engine,
            "recognizer_version",
            config.get("calibration.recognizer_version")
            or config.get("calibration.arcface_model_version", "buffalo_l"),
        )
    )
    licence = str(getattr(engine, "licence", ""))
    cal_recognizer_version = (
        evaluator.recognizer_version()
        if hasattr(evaluator, "recognizer_version")
        else evaluator.arcface_model_version()
    )
    tpl_recognizer_version = (
        template_evaluator.recognizer_version()
        if hasattr(template_evaluator, "recognizer_version")
        else template_evaluator.arcface_model_version()
    )

    return {
        "status": "ok",
        "inference_backend": "triton",
        "triton_endpoint": engine.endpoint,
        "model_profile": engine.model_profile,
        "active_profile": active_profile_name,
        "recognizer_version": recognizer_version,
        "licence": licence,
        "enrolled_members": await store.member_count(),
        "gallery_representation": getattr(store, "gallery_representation_mode", "centroid"),
        "active_gallery_representation": getattr(store, "active_representation", "centroid"),
        "model": recognizer_version,
        "models": engine.model_names,
        "calibration": {
            "status": cal_state,
            "artifact_version": evaluator.artifact_version(),
            "recognizer_version": cal_recognizer_version,
            "arcface_model_version": cal_recognizer_version,
            "preprocessing_version": evaluator.preprocessing_version(),
            "profile": evaluator.calibration_profile(),
            "quality_version": evaluator.quality_version(),
            "gallery_representation": evaluator.gallery_representation(),
            "covered_identities": evaluator.covered_identities(),
        },
        # Calibration for /match-template templates: its own artifact, profile, and coverage.
        "template_calibration": {
            "status": template_evaluator.health(),
            "artifact_version": template_evaluator.artifact_version(),
            "recognizer_version": tpl_recognizer_version,
            "profile": template_evaluator.calibration_profile(),
            "aggregation": template_evaluator.aggregation(),
            "quality_version": template_evaluator.quality_version(),
            "gallery_representation": template_evaluator.gallery_representation(),
            "covered_identities": template_evaluator.covered_identities(),
        },
    }


@router.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> Response:
    """Expose the default prometheus_client registry in the Prometheus text format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
