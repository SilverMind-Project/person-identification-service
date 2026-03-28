"""Pydantic models for standalone motion detection endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MotionDetectionRequest(BaseModel):
    images: list[str] = Field(
        ..., min_length=2, description="Ordered list of base64-encoded images (at least 2)"
    )


class TrajectoryPoint(BaseModel):
    cx: float
    cy: float
    width: float
    height: float


class PersonTrack(BaseModel):
    track_id: int
    person_id: str
    name: str
    direction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    trajectory: list[TrajectoryPoint]


class MotionDetectionResponse(BaseModel):
    persons: list[PersonTrack]
