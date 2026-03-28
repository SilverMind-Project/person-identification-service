"""Face identification endpoints."""

from __future__ import annotations

import base64
import logging

import cv2
from fastapi import APIRouter, Request

from app.models.identification import (
    BatchIdentifyRequest,
    BatchIdentifyResponse,
    FaceDetection,
    FrameResult,
    IdentifyRequest,
    IdentifyResponse,
    PersonMotion,
)
from app.services.face_engine import decode_base64_image
from app.services.image_annotator import annotate_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["identification"])


def _encode_image_to_base64(image) -> str:
    """Encode a BGR numpy array to a base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


@router.post("/identify", response_model=IdentifyResponse)
async def identify(request: Request, body: IdentifyRequest):
    """Identify faces in a single image."""
    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store

    image = decode_base64_image(body.image)
    faces = engine.detect_faces(image)
    identities = store.identify_all(faces)

    # Save the full image if any unidentified guests are present
    if body.save_guest_images:
        guest_count = sum(1 for r in identities if r.person_id == "unknown")
        if guest_count > 0:
            guest_store = request.app.state.guest_store
            guest_store.save_guest_image(image, guest_count=guest_count)

    annotated_b64 = None
    if body.include_annotated_image and identities:
        annotated = annotate_image(image, identities)
        annotated_b64 = _encode_image_to_base64(annotated)

    return IdentifyResponse(
        faces=[
            FaceDetection(
                person_id=r.person_id,
                name=r.name,
                confidence=r.confidence,
                bbox=r.bbox,
            )
            for r in identities
        ],
        annotated_image=annotated_b64,
    )


@router.post("/identify-batch", response_model=BatchIdentifyResponse)
async def identify_batch(request: Request, body: BatchIdentifyRequest):
    """Identify faces across a batch of images, optionally computing motion direction.

    This is the primary endpoint consumed by the Cognitive Companion v2 backend.
    It accepts the full batch from the event aggregator (typically 5 frames).
    """
    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store
    motion_detector = request.app.state.motion_detector

    frames: list[FrameResult] = []
    all_faces = []
    all_identities = []
    frame_shapes = []
    decoded_images = []

    for idx, b64_image in enumerate(body.images):
        try:
            image = decode_base64_image(b64_image)
        except ValueError:
            logger.warning("Failed to decode image at frame index %d", idx)
            frames.append(FrameResult(frame_index=idx, faces=[]))
            all_faces.append([])
            all_identities.append([])
            frame_shapes.append((0, 0))
            decoded_images.append(None)
            continue

        decoded_images.append(image)
        frame_shapes.append((image.shape[0], image.shape[1]))
        faces = engine.detect_faces(image)
        identities = store.identify_all(faces)
        all_faces.append(faces)
        all_identities.append(identities)

        # Save the full image if any unidentified guests are present
        if body.save_guest_images:
            guest_count = sum(1 for r in identities if r.person_id == "unknown")
            if guest_count > 0:
                guest_store = request.app.state.guest_store
                guest_store.save_guest_image(image, guest_count=guest_count, frame_index=idx)

        frames.append(
            FrameResult(
                frame_index=idx,
                faces=[
                    FaceDetection(
                        person_id=r.person_id,
                        name=r.name,
                        confidence=r.confidence,
                        bbox=r.bbox,
                    )
                    for r in identities
                ],
            )
        )

    # Motion detection
    motion: list[PersonMotion] = []
    if body.include_motion and len(body.images) >= 2:
        tracks = motion_detector.detect_direction(
            frame_shapes=frame_shapes,
            frame_faces=all_faces,
            frame_identities=all_identities,
        )
        motion = [
            PersonMotion(
                person_id=t.person_id,
                name=t.name,
                direction=t.direction,
                confidence=t.confidence,
            )
            for t in tracks
        ]

    # Annotated images
    annotated_images: list[str] | None = None
    if body.include_annotated_image:
        annotated_images = []
        for idx, image in enumerate(decoded_images):
            if image is None:
                annotated_images.append("")
                continue
            identities = all_identities[idx] if idx < len(all_identities) else []
            if identities:
                annotated = annotate_image(image, identities)
                annotated_images.append(_encode_image_to_base64(annotated))
            else:
                annotated_images.append(_encode_image_to_base64(image))

    return BatchIdentifyResponse(
        frames=frames,
        motion=motion,
        annotated_images=annotated_images,
    )
