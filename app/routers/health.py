"""Health check endpoint."""

from fastapi import APIRouter, Request

from app import config

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store
    return {
        "status": "ok",
        "inference_backend": "triton",
        "triton_endpoint": engine.endpoint,
        "model_profile": engine.model_profile,
        "enrolled_members": await store.member_count(),
        "model": config.get("face_engine.model_name", "buffalo_l"),
        "models": engine.model_names,
    }
