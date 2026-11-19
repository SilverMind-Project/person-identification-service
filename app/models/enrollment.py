"""Pydantic models for enrollment and face-prototype endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

PoseBinLiteral = Literal["frontal", "left", "right", "down"]
EmbeddingSourceLiteral = Literal[
    "enrollment_photo",
    "deployment_capture",
    "visitor_naming",
    "wearable_exemplar",
]


class EnrollRequest(BaseModel):
    person_id: str = Field(
        ..., min_length=1, max_length=64, description="Unique identifier for the person"
    )
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
    gallery_revision: int = Field(
        default=1,
        ge=1,
        description="Monotonic revision incremented whenever this member's gallery changes",
    )
    prototype_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Prototype count per pose bin (frontal, left, right, down)",
    )


class MemberListResponse(BaseModel):
    members: list[MemberInfo]
    total: int


class PrototypeInfo(BaseModel):
    prototype_id: str
    person_id: str
    pose_bin: PoseBinLiteral
    member_count: int = Field(..., ge=1)
    source: EmbeddingSourceLiteral
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_version: str | None = None
    crop_object_key: str | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    created_at: datetime | None = None


class MemberPrototypesResponse(BaseModel):
    person_id: str
    gallery_revision: int = Field(..., ge=1)
    prototypes: list[PrototypeInfo]
    counts_by_bin: dict[str, int]


class RebuildPrototypesResponse(BaseModel):
    person_id: str
    prototype_count: int = Field(..., ge=0)
    counts_by_bin: dict[str, int]
