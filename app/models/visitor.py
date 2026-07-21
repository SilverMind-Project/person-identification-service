"""Pydantic models for visitor cluster review endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class VisitorSighting(BaseModel):
    seen_at: datetime
    quality: float
    crop_object: str | None = None


class VisitorClusterSummary(BaseModel):
    cluster_id: str
    status: str = Field(..., description="candidate | surfaced | named | dismissed")
    display_hint: str | None = None
    named_person_id: str | None = None
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
