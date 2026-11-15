"""Pydantic models for visitor cluster review endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VisitorSighting(BaseModel):
    seen_at: datetime
    quality: float
    crop_object: str | None = None


class VisitorClusterSummary(BaseModel):
    cluster_id: str
    status: str = Field(..., description="candidate | surfaced | named | dismissed | attributed")
    display_hint: str | None = None
    named_person_id: str | None = None
    attributed_person_id: str | None = Field(
        default=None,
        description="The enrolled member an attributed cluster belongs to; never enrolled from it",
    )
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    sighting_count: int
    distinct_days: int
    first_seen_at: datetime
    last_seen_at: datetime
    recent_crop_keys: list[str] = Field(default_factory=list)


class VisitorClusterDetail(VisitorClusterSummary):
    recent_sightings: list[VisitorSighting] = Field(default_factory=list)


class VisitorClusterListResponse(BaseModel):
    clusters: list[VisitorClusterSummary]
    total: int


class NameClusterRequest(BaseModel):
    person_id: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=128)


class NameClusterResponse(BaseModel):
    cluster_id: str
    status: str
    named_person_id: str
    member_name: str
    embedding_count: int


class ReviewClustersRequest(BaseModel):
    """Attribute clusters to an enrolled member, or dismiss them, in one call."""

    model_config = {"extra": "forbid"}

    cluster_ids: list[str] = Field(..., min_length=1, max_length=200)
    verdict: Literal["attribute", "dismiss"]
    person_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Required for attribute: an enrolled member. Nothing is enrolled.",
    )
    actor: str = Field(..., min_length=1, max_length=128, description="Who decided")


class ClusterReviewResult(BaseModel):
    cluster_id: str
    ok: bool
    error_code: Literal["not_found", "already_named", "unknown_member"] | None = None


class ReviewClustersResponse(BaseModel):
    results: list[ClusterReviewResult]
    reviewed: int
    failed: int
