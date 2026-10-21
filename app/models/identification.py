"""Pydantic models for identification endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field, field_validator


class FaceQualityModel(BaseModel):
    """Face image quality and the components it was combined from, all in [0, 1].

    ``version`` names the scorer; calibration artifacts and downstream policy bind to it.
    """

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Combined quality (weighted geometric mean)"
    )
    resolution: float = Field(..., ge=0.0, le=1.0, description="Interocular-distance component")
    sharpness: float = Field(
        ..., ge=0.0, le=1.0, description="Laplacian-variance component, on the aligned crop"
    )
    detection: float = Field(..., ge=0.0, le=1.0, description="SCRFD detection-score component")
    pitch: float = Field(..., ge=0.0, le=1.0, description="Head-pitch component (yaw is excluded)")
    version: str = Field(..., description="Scorer version, for example heuristic-v1")


class FaceDetection(BaseModel):
    person_id: str
    name: str
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] bounding box")
    recognition_state: str = Field(
        default="recognized",
        description="One of: recognized, candidate, unrecognized",
    )
    best_candidate_id: str | None = Field(
        default=None, description="Nearest centroid person_id, even when below threshold"
    )
    similarity: float = Field(
        default=0.0,
        description=(
            "Raw ArcFace cosine similarity to the best candidate centroid; the face's match score. "
            "Not a probability: calibrated_confidence is the probability."
        ),
    )
    yaw_deg: float = Field(default=0.0, description="Head pose yaw in degrees")
    pitch_deg: float = Field(default=0.0, description="Head pose pitch in degrees")
    roll_deg: float = Field(default=0.0, description="Head pose roll in degrees")
    det_score: float = Field(default=0.0, description="SCRFD detection confidence score")
    face_width_px: float = Field(
        default=0.0, ge=0.0, description="Face box width in pixels of the submitted image"
    )
    face_height_px: float = Field(
        default=0.0, ge=0.0, description="Face box height in pixels of the submitted image"
    )
    interocular_px: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Distance between the two SCRFD eye keypoints in pixels of the submitted image; "
            "null when the keypoints are unavailable"
        ),
    )
    quality: FaceQualityModel | None = Field(
        default=None, description="Face image quality; null when the face was not scored"
    )
    # ArcFace calibration fields
    calibrated_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated match probability from logistic or isotonic regression on ArcFace similarity; null when calibration artifact is missing, incompatible, or invalid",
    )
    calibration_status: str = Field(
        default="degraded_missing",
        description="ready | degraded_missing | degraded_incompatible | degraded_invalid",
    )
    calibration_artifact_version: str | None = Field(default=None)
    calibration_profile: str = Field(
        default="",
        description=(
            "Profile of the calibration that produced calibrated_confidence: single_frame for "
            "this endpoint's per-face calibration; empty when no artifact is loaded"
        ),
    )
    arcface_model_version: str = Field(default="")
    model_profile: str = Field(default="")
    preprocessing_version: str = Field(default="")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_similarity(self) -> float:
        """Alias of ``similarity``, kept for one release for older consumers."""
        return self.similarity


class CaptureContext(BaseModel):
    """Where and how the caller produced the image; sent by tracking-orchestrator.

    When present and ``calibration_capture.enabled`` is on, faces from this call may be kept as
    calibration candidates (aligned crop, embedding, quality) for operator labeling.
    """

    camera_id: str = Field(..., min_length=1, max_length=128)
    captured_at: datetime = Field(..., description="Frame capture time (timezone-aware)")
    crop_pad_top_frac: float = Field(..., ge=0.0, le=1.0)
    crop_pad_side_frac: float = Field(..., ge=0.0, le=1.0)
    crop_jpeg_quality: int = Field(..., ge=1, le=100)

    @field_validator("captured_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware")
        return value


class IdentifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    capture_context: CaptureContext | None = Field(
        default=None,
        description=(
            "Capture provenance. Side effect: with calibration capture enabled, faces from this "
            "call may be stored as calibration candidates."
        ),
    )
    include_annotated_image: bool = Field(
        default=False, description="Return image with bounding boxes and labels drawn"
    )
    save_guest_images: bool = Field(
        default=False, description="Save images if unidentified guests are detected"
    )


class IdentifyResponse(BaseModel):
    faces: list[FaceDetection]
    image_width_px: int = Field(default=0, ge=0, description="Decoded input image width")
    image_height_px: int = Field(default=0, ge=0, description="Decoded input image height")
    annotated_image: str | None = Field(
        default=None, description="Base64-encoded annotated image (if requested)"
    )


class BatchIdentifyRequest(BaseModel):
    images: list[str] = Field(
        ..., min_length=1, description="Ordered list of base64-encoded images"
    )
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
    image_width_px: int = Field(
        default=0, ge=0, description="Decoded frame width; 0 if undecodable"
    )
    image_height_px: int = Field(
        default=0, ge=0, description="Decoded frame height; 0 if undecodable"
    )


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
