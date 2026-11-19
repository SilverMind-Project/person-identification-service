"""Enrollment endpoints for managing household members."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.models.enrollment import (
    EnrollRequest,
    EnrollResult,
    MemberInfo,
    MemberListResponse,
    MemberPrototypesResponse,
    PrototypeInfo,
    RebuildPrototypesResponse,
)
from app.services.face_engine import decode_base64_image
from app.services.prototype_builder import Prototype

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["enrollment"])


@router.post("/enroll", response_model=EnrollResult)
async def enroll(request: Request, body: EnrollRequest):
    """Enroll a new household member with face images."""
    logger.info(
        "POST /enroll person_id=%s name=%s images=%d",
        body.person_id,
        body.name,
        len(body.images),
    )

    store = request.app.state.enrollment_store

    images: list[np.ndarray] = []
    for idx, b64 in enumerate(body.images):
        try:
            images.append(decode_base64_image(b64))
        except ValueError:
            logger.warning("Failed to decode image at index %d for person %s", idx, body.person_id)

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    result = await store.enroll(body.person_id, body.name, images)
    if result.status == "failed":
        logger.warning(
            "POST /enroll -> failed person_id=%s reason=no_faces_detected",
            body.person_id,
        )
        raise HTTPException(
            status_code=422,
            detail=f"No faces detected in any of the {len(body.images)} images",
        )
    logger.info(
        "POST /enroll -> person_id=%s status=%s embedding_count=%d failed_images=%s",
        result.person_id,
        result.status,
        result.embedding_count,
        result.failed_images,
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
    logger.info(
        "POST /enroll/upload person_id=%s name=%s files=%d",
        person_id,
        name,
        len(files),
    )

    store = request.app.state.enrollment_store
    images: list[np.ndarray] = []
    for f in files:
        data = await f.read()
        np_arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        raise HTTPException(status_code=400, detail="No valid image files provided")

    result = await store.enroll(person_id, name, images)
    if result.status == "failed":
        logger.warning(
            "POST /enroll/upload -> failed person_id=%s reason=no_faces_detected",
            person_id,
        )
        raise HTTPException(status_code=422, detail="No faces detected in any uploaded images")
    logger.info(
        "POST /enroll/upload -> person_id=%s status=%s embedding_count=%d failed_images=%s",
        result.person_id,
        result.status,
        result.embedding_count,
        result.failed_images,
    )
    return result


@router.get("/members", response_model=MemberListResponse)
async def list_members(request: Request):
    """List all enrolled household members."""
    store = request.app.state.enrollment_store
    members = await store.list_members()
    return MemberListResponse(members=members, total=len(members))


@router.get("/members/{person_id}", response_model=MemberInfo)
async def get_member(request: Request, person_id: str):
    """Get details of a specific enrolled member."""
    store = request.app.state.enrollment_store
    member = await store.get_member(person_id)
    if not member:
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    return member


@router.delete("/members/{person_id}")
async def delete_member(request: Request, person_id: str):
    """Remove an enrolled member and all their embeddings."""
    store = request.app.state.enrollment_store
    if not await store.remove_member(person_id):
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    return {"deleted": True, "person_id": person_id}


def _proto_info(proto: Prototype) -> PrototypeInfo:
    return PrototypeInfo(
        prototype_id=proto.prototype_id,
        person_id=proto.person_id,
        pose_bin=proto.pose_bin,
        member_count=proto.member_count,
        source=proto.source,
        quality_score=proto.quality_score,
        quality_version=proto.quality_version,
        crop_object_key=proto.crop_object_key,
        approved_by=proto.approved_by,
        approved_at=proto.approved_at,
        created_at=proto.created_at,
    )


def _counts_by_bin(prototypes: list[Prototype]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for proto in prototypes:
        counts[proto.pose_bin] = counts.get(proto.pose_bin, 0) + 1
    return dict(sorted(counts.items()))


@router.get("/members/{person_id}/prototypes", response_model=MemberPrototypesResponse)
async def list_member_prototypes(request: Request, person_id: str) -> MemberPrototypesResponse:
    """List pose-binned prototypes for one enrolled member."""
    store = request.app.state.enrollment_store
    member = await store.get_member(person_id)
    if member is None:
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    prototypes = await store.list_prototypes(person_id)
    return MemberPrototypesResponse(
        person_id=person_id,
        gallery_revision=member.gallery_revision,
        prototypes=[_proto_info(p) for p in prototypes],
        counts_by_bin=_counts_by_bin(prototypes),
    )


@router.post("/members/{person_id}/prototypes/rebuild", response_model=RebuildPrototypesResponse)
async def rebuild_member_prototypes_route(
    request: Request, person_id: str
) -> RebuildPrototypesResponse:
    """Rebuild pose-binned prototypes for one enrolled member from all stored embeddings."""
    store = request.app.state.enrollment_store
    member = await store.get_member(person_id)
    if member is None:
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    prototypes = await store.rebuild_prototypes(person_id)
    return RebuildPrototypesResponse(
        person_id=person_id,
        prototype_count=len(prototypes),
        counts_by_bin=_counts_by_bin(prototypes),
    )


@router.delete("/members/{person_id}/prototypes/{prototype_id}")
async def delete_member_prototype(
    request: Request, person_id: str, prototype_id: str
) -> dict[str, object]:
    """Delete one prototype (and its underlying embedding(s)), recompute centroid and prototypes.

    Refuses with 409 when removal would leave the member with zero embeddings; delete the member's
    enrollment instead.
    """
    store = request.app.state.enrollment_store
    member = await store.get_member(person_id)
    if member is None:
        raise HTTPException(status_code=404, detail=f"Member '{person_id}' not found")
    try:
        deleted = await store.delete_prototype(person_id, prototype_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Prototype '{prototype_id}' not found")
    return {"deleted": True, "person_id": person_id, "prototype_id": prototype_id}
