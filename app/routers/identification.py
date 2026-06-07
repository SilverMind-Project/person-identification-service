"""Face identification endpoints."""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np
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
    logger.info(
        "POST /identify image_size=%d include_annotated=%s save_guests=%s",
        len(body.image),
        body.include_annotated_image,
        body.save_guest_images,
    )

    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store

    image = decode_base64_image(body.image)
    faces = await engine.detect_faces(image)
    identities = await store.identify_all(faces)

    # Save the full image if any unidentified guests are present
    if body.save_guest_images:
        guest_count = sum(1 for r in identities if r.person_id == "unknown")
        if guest_count > 0:
            guest_store = request.app.state.guest_store
            await guest_store.save_guest_image(image, guest_count=guest_count)

    annotated_b64 = None
    if body.include_annotated_image and identities:
        annotated = annotate_image(image, identities)
        annotated_b64 = _encode_image_to_base64(annotated)

    response = IdentifyResponse(
        faces=[
            FaceDetection(
                person_id=r.person_id,
                name=r.name,
                confidence=r.confidence,
                bbox=r.bbox,
                recognition_state=r.recognition_state,
                best_candidate_id=r.best_candidate_id,
                similarity=r.similarity,
                yaw_deg=r.yaw_deg,
                pitch_deg=r.pitch_deg,
                roll_deg=r.roll_deg,
                det_score=r.det_score,
            )
            for r in identities
        ],
        annotated_image=annotated_b64,
    )
    logger.info(
        "POST /identify -> faces=%d persons=[%s] annotated_size=%s",
        len(response.faces),
        ", ".join(f.person_id for f in response.faces) if response.faces else "(none)",
        len(annotated_b64) if annotated_b64 else "none",
    )
    return response


@router.post("/identify-batch", response_model=BatchIdentifyResponse)
async def identify_batch(request: Request, body: BatchIdentifyRequest):
    """Identify faces across a batch of images, optionally computing motion direction.

    This is the primary endpoint consumed by the Cognitive Companion v2 backend.
    It accepts the full batch from the event aggregator (typically 5 frames).
    """
    logger.info(
        "POST /identify-batch images=%d include_motion=%s include_annotated=%s save_guests=%s",
        len(body.images),
        body.include_motion,
        body.include_annotated_image,
        body.save_guest_images,
    )

    engine = request.app.state.face_engine
    store = request.app.state.enrollment_store
    motion_detector = request.app.state.motion_detector

    frames: list[FrameResult] = []
    all_faces: list[list] = []
    all_identities: list[list] = []
    frame_shapes: list[tuple[int, int]] = []
    decoded_images: list[np.ndarray | None] = []

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
        faces = await engine.detect_faces(image)
        identities = await store.identify_all(faces)
        all_faces.append(faces)
        all_identities.append(identities)

        # Save the full image if any unidentified guests are present
        if body.save_guest_images:
            guest_count = sum(1 for r in identities if r.person_id == "unknown")
            if guest_count > 0:
                guest_store = request.app.state.guest_store
                await guest_store.save_guest_image(image, guest_count=guest_count, frame_index=idx)

        frames.append(
            FrameResult(
                frame_index=idx,
                faces=[
                    FaceDetection(
                        person_id=r.person_id,
                        name=r.name,
                        confidence=r.confidence,
                        bbox=r.bbox,
                        recognition_state=r.recognition_state,
                        best_candidate_id=r.best_candidate_id,
                        similarity=r.similarity,
                        yaw_deg=r.yaw_deg,
                        pitch_deg=r.pitch_deg,
                        roll_deg=r.roll_deg,
                        det_score=r.det_score,
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
        for idx, decoded in enumerate(decoded_images):
            if decoded is None:
                annotated_images.append("")
                continue
            identities = all_identities[idx] if idx < len(all_identities) else []
            if identities:
                annotated = annotate_image(decoded, identities)
                annotated_images.append(_encode_image_to_base64(annotated))
            else:
                annotated_images.append(_encode_image_to_base64(decoded))

    response = BatchIdentifyResponse(
        frames=frames,
        motion=motion,
        annotated_images=annotated_images,
    )
    face_counts = [len(f.faces) for f in response.frames]
    logger.info(
        "POST /identify-batch -> frames=%d faces_per_frame=%s motion_tracks=%d annotated=%d",
        len(response.frames),
        face_counts,
        len(response.motion),
        len(annotated_images) if annotated_images else 0,
    )
    return response
