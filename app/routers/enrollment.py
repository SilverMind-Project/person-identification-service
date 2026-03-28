"""Enrollment endpoints for managing household members."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form

from app.models.enrollment import EnrollRequest, EnrollResult, MemberInfo, MemberListResponse
from app.services.face_engine import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["enrollment"])


@router.post("/enroll", response_model=EnrollResult)
async def enroll(request: Request, body: EnrollRequest):
    """Enroll a new household member with face images."""
    store = request.app.state.enrollment_store

    images: list[np.ndarray] = []
    for idx, b64 in enumerate(body.images):
        try:
            images.append(decode_base64_image(b64))
        except ValueError:
            logger.warning("Failed to decode image at index %d for person %s", idx, body.person_id)

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    result = store.enroll(body.person_id, body.name, images)
    if result.status == "failed":
        raise HTTPException(
            status_code=422,
            detail=f"No faces detected in any of the {len(body.images)} images",
        )
    return result


@router.post("/enroll/upload/{person_id}", response_model=EnrollResult)
async def enroll_upload(
    request: Request,
    person_id: str,
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Enroll via multipart file upload (convenience for curl/admin tools)."""
    store = request.app.state.enrollment_store
    import cv2

    images: list[np.ndarray] = []
    for f in files:
        data = await f.read()
        np_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        raise HTTPException(status_code=400, detail="No valid image files provided")

    result = store.enroll(person_id, name, images)
    if result.status == "failed":
        raise HTTPException(
            status_code=422, detail="No faces detected in any uploaded images"
        )
    return result


@router.get("/members", response_model=MemberListResponse)
async def list_members(request: Request):
    """List all enrolled household members."""
    store = request.app.state.enrollment_store
    members = store.list_members()
    return MemberListResponse(members=members, total=len(members))


@router.get("/members/{person_id}", response_model=MemberInfo)
async def get_member(request: Request, person_id: str):
    """Get details of a specific enrolled member."""
    store = request.app.state.enrollment_store
    member = store.get_member(person_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    return member


@router.delete("/members/{person_id}")
async def delete_member(request: Request, person_id: str):
    """Remove an enrolled member and all their embeddings."""
    store = request.app.state.enrollment_store
    if not store.remove_member(person_id):
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    return {"deleted": True, "person_id": person_id}
