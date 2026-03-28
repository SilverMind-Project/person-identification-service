"""Pydantic models for identification endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceDetection(BaseModel):
    person_id: str
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] bounding box")


class IdentifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    include_annotated_image: bool = Field(
        default=False, description="Return image with bounding boxes and labels drawn"
    )
    save_guest_images: bool = Field(
        default=False, description="Save images if unidentified guests are detected"
    )


class IdentifyResponse(BaseModel):
    faces: list[FaceDetection]
    annotated_image: str | None = Field(
        default=None, description="Base64-encoded annotated image (if requested)"
    )


class BatchIdentifyRequest(BaseModel):
    images: list[str] = Field(..., min_length=1, description="Ordered list of base64-encoded images")
    include_motion: bool = Field(default=True, description="Compute motion direction across frames")
    include_annotated_image: bool = Field(
        default=False, description="Return annotated images with bounding boxes and labels"
    )
    save_guest_images: bool = Field(
        default=False, description="Save images if unidentified guests are detected"
    )


class FrameResult(BaseModel):
    frame_index: int
    faces: list[FaceDetection]


class PersonMotion(BaseModel):
    person_id: str
    name: str
    direction: str = Field(
        ...,
        description="Movement direction: left-to-right, right-to-left, towards-camera, away-from-camera, stationary",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchIdentifyResponse(BaseModel):
    frames: list[FrameResult]
    motion: list[PersonMotion] = Field(default_factory=list)
    annotated_images: list[str] | None = Field(
        default=None,
        description="Base64-encoded annotated images, one per frame (if requested)",
    )
