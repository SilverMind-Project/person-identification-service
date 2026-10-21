"""Pydantic models for the calibration-candidate review endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CandidateView(BaseModel):
    """A capture as the review screen sees it: no embedding, a crop key the caller presigns."""

    candidate_id: str
    captured_at: datetime
    camera_id: str
    crop_object_key: str
    quality_score: float
    quality_version: str
    interocular_px: float | None
    yaw_deg: float
    pitch_deg: float
    nearest_person_id: str | None
    nearest_similarity: float = Field(
        ..., description="Raw cosine similarity to the nearest member"
    )
    label_state: Literal["unlabeled", "labeled", "rejected"]
    label_kind: Literal["member", "not_enrolled", "not_a_face"] | None = None
    label_identity_id: str | None = None
    labeled_by: str | None = None
    labeled_at: datetime | None = None


class CandidateListResponse(BaseModel):
    candidates: list[CandidateView]
    next_cursor: str | None = None


class LabelRequest(BaseModel):
    label_kind: Literal["member", "not_enrolled", "not_a_face"]
    label_identity_id: str | None = Field(default=None, max_length=64)
    labeled_by: str = Field(..., min_length=1, max_length=128)

    @model_validator(mode="after")
    def _identity_only_for_members(self) -> LabelRequest:
        if (self.label_kind == "member") != (self.label_identity_id is not None):
            raise ValueError("label_identity_id is required for member labels and only for them")
        return self


class IdentityProgress(BaseModel):
    identity_id: str
    labeled_faces: int
    episodes: int
    # Labeled faces needed so the household holdout (about 20% of episodes) can reach the
    # per-identity genuine minimum; the fitter is the authority, this is a progress hint.
    target_labeled_faces: int


class CalibrationSummary(BaseModel):
    by_state: dict[str, int]
    by_label_kind: dict[str, int]
    by_quality_decile: dict[str, int]
    by_camera: dict[str, int]
    identities: list[IdentityProgress]
    not_enrolled_episodes: int
    capture_enabled: bool
