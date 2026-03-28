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
        "gpu_available": engine.gpu_available,
        "enrolled_members": store.member_count,
        "model": config.get("face_engine.model_name", "buffalo_l"),
    }
