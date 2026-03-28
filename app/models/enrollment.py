"""Pydantic models for enrollment endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EnrollRequest(BaseModel):
    person_id: str = Field(..., min_length=1, max_length=64, description="Unique identifier for the person")
    name: str = Field(..., min_length=1, max_length=128, description="Display name")
    images: list[str] = Field(
        ...,
        min_length=1,
        description="List of base64-encoded face images (JPEG/PNG). Minimum 1, recommended 5-10.",
    )


class EnrollResult(BaseModel):
    person_id: str
    name: str
    embedding_count: int
    status: str  # "enrolled" | "updated"
    failed_images: list[int] = Field(
        default_factory=list,
        description="Indices of images where no face was detected",
    )


class MemberInfo(BaseModel):
    person_id: str
    name: str
    embedding_count: int
    created_at: datetime


class MemberListResponse(BaseModel):
    members: list[MemberInfo]
    total: int
