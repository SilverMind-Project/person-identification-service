"""Pydantic models for template matching (several faces of one person matched together)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.identification import FaceQualityModel

# Fewer than two faces is a single-face identify; more than sixteen adds latency for no gain once
# the quality-weighted mean has settled.
MIN_TEMPLATE_MEMBERS = 2
MAX_TEMPLATE_MEMBERS = 16


class TemplateMemberModel(BaseModel):
    embedding_f16_b64: str = Field(
        ...,
        min_length=1,
        description="512 little-endian float16 values, base64, from /identify return_embedding",
    )
    quality: FaceQualityModel = Field(
        ..., description="The face's quality object from the same /identify response"
    )


class MatchTemplateRequest(BaseModel):
    members: list[TemplateMemberModel] = Field(
        ..., min_length=MIN_TEMPLATE_MEMBERS, max_length=MAX_TEMPLATE_MEMBERS
    )
    embedding_model_version: str = Field(
        ..., min_length=1, description="ArcFace model version the embeddings came from"
    )


class TemplateQualityModel(BaseModel):
    """Template quality features: means over the best three members by quality score."""

    score: float = Field(..., ge=0.0, le=1.0)
    resolution: float = Field(..., ge=0.0, le=1.0)
    sharpness: float = Field(..., ge=0.0, le=1.0)
    detection: float = Field(..., ge=0.0, le=1.0)
    pitch: float = Field(..., ge=0.0, le=1.0)


class MatchTemplateResponse(BaseModel):
    person_id: str
    name: str
    recognition_state: str = Field(..., description="recognized, candidate, or unrecognized")
    best_candidate_id: str | None = None
    raw_similarity: float = Field(
        ...,
        description="Raw cosine similarity of the template to the nearest centroid; not a probability",
    )
    second_best_similarity: float | None = Field(
        default=None,
        description="Raw cosine similarity to the second-nearest centroid; null for one member",
    )
    member_count: int = Field(..., ge=MIN_TEMPLATE_MEMBERS, le=MAX_TEMPLATE_MEMBERS)
    aggregator_version: str
    template_quality: TemplateQualityModel
    quality_version: str = Field(..., description="Face-quality scorer version of the members")
    calibrated_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated match probability from the template-profile artifact; null for a "
            "template that is not recognized, without a compatible template artifact, or for an "
            "identity the artifact does not cover"
        ),
    )
    calibration_profile: str = Field(
        default="", description="template when a template artifact is loaded; empty otherwise"
    )
    calibration_status: str = Field(default="degraded_missing")
    calibration_artifact_version: str | None = None
