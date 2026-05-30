"""Standalone motion direction detection endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.models.motion import (
    MotionDetectionRequest,
    MotionDetectionResponse,
    PersonTrack,
    TrajectoryPoint,
)
from app.services.face_engine import decode_base64_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["motion"])


@router.post("/detect-motion", response_model=MotionDetectionResponse)
async def detect_motion(request: Request, body: MotionDetectionRequest):
    """Detect direction of motion for persons across a frame sequence."""
    logger.info("POST /detect-motion images=%d", len(body.images))

    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store
    motion_detector = request.app.state.motion_detector

    all_faces: list[list] = []
    all_identities: list[list] = []
    frame_shapes: list[tuple[int, int]] = []

    for idx, b64_image in enumerate(body.images):
        try:
            image = decode_base64_image(b64_image)
        except ValueError:
            logger.warning("Failed to decode image at frame index %d", idx)
            all_faces.append([])
            all_identities.append([])
            frame_shapes.append((0, 0))
            continue

        frame_shapes.append((image.shape[0], image.shape[1]))
        faces = engine.detect_faces(image)
        identities = await store.identify_all(faces)
        all_faces.append(faces)
        all_identities.append(identities)

    tracks = motion_detector.detect_direction(
        frame_shapes=frame_shapes,
        frame_faces=all_faces,
        frame_identities=all_identities,
    )

    response = MotionDetectionResponse(
        persons=[
            PersonTrack(
                track_id=t.track_id,
                person_id=t.person_id,
                name=t.name,
                direction=t.direction,
                confidence=t.confidence,
                trajectory=[
                    TrajectoryPoint(cx=p.cx, cy=p.cy, width=p.width, height=p.height)
                    for p in t.trajectory
                ],
            )
            for t in tracks
        ]
    )
    logger.info(
        "POST /detect-motion -> tracks=%d persons=[%s]",
        len(response.persons),
        ", ".join(f"{p.person_id}:{p.direction}" for p in response.persons)
        if response.persons
        else "(none)",
    )
    return response
