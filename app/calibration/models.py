"""Pydantic models for ArcFace calibration artifacts.

The artifact is stored as versioned JSON (never pickle/joblib) so it is
safe to load from untrusted storage and auditable in version control.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SplitCounts(BaseModel):
    """Pair and identity counts for each data partition."""

    fit_pairs: int = 0
    fit_positives: int = 0
    fit_negatives: int = 0
    val_pairs: int = 0
    val_positives: int = 0
    val_negatives: int = 0
    test_pairs: int = 0
    test_positives: int = 0
    test_negatives: int = 0
    fit_identities: int = 0
    val_identities: int = 0
    test_identities: int = 0


class CalibrationMetrics(BaseModel):
    """Evaluation metrics recorded at fit time (test split)."""

    brier_score: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    fmr_at_operating_point: float | None = None
    fnmr_at_operating_point: float | None = None
    operating_point_threshold: float | None = None


class CalibrationManifest(BaseModel):
    """Metadata fields shared between artifact and manifest files."""

    schema_version: int = 1
    artifact_version: str
    method: Literal["logistic", "isotonic"]
    arcface_model_version: str
    model_profile: str
    preprocessing_version: str
    fitted_at: datetime
    random_seed: int = 42
    tool_version: str = ""
    fit_dataset_hash: str = ""
    val_dataset_hash: str = ""
    test_dataset_hash: str = ""
    split_counts: SplitCounts = Field(default_factory=SplitCounts)
    metrics: CalibrationMetrics = Field(default_factory=CalibrationMetrics)


class CalibrationArtifact(CalibrationManifest):
    """Complete calibration artifact including fitted parameters.

    Logistic regression: single-feature Platt-style scaling.
        P(match | x) = sigmoid(coef * x + intercept)

    Isotonic regression: piecewise-linear interpolation over knots.
        P(match | x) = interp(x_knots, y_knots, x)  (clamped at boundaries)
    """

    coef: float | None = None
    intercept: float | None = None
    x_knots: list[float] | None = None
    y_knots: list[float] | None = None


FEATURE_NAMES = (
    "similarity",
    "quality_score",
    "resolution",
    "sharpness",
    "detection",
    "pitch",
    "margin",
)

GalleryRepresentation = Literal["centroid", "prototypes-v1"]


class ExternalDatasetRef(BaseModel):
    """An external face set pooled into the fit; never part of a gate."""

    name: str
    version: str
    licence: str = Field(..., min_length=1)
    identities: int = 0
    images: int = 0
    images_without_face: int = 0


class CalibrationArtifactV2(CalibrationManifest):
    """Schema 2: multi-feature calibration fitted on this household's own captures.

    ``logistic_v2``: P(match | x) = sigmoid(intercept + sum(coefficients[i] * x[features[i]])), with
    coefficients in raw feature space so the runtime needs no scaler. ``isotonic`` keeps one
    feature (``similarity``). The runtime returns no calibrated confidence for any identity outside
    ``covered_identities`` (fail closed per identity: someone enrolled after the fit, or with too
    few holdout faces, has to be recalibrated first).
    """

    schema_version: Literal[2] = 2
    method: Literal["logistic_v2", "isotonic"]  # type: ignore[assignment]
    profile: Literal["single_frame", "template"] = "single_frame"
    quality_version: str = Field(..., min_length=1)
    # The template aggregator the scores were fitted with (``qwmean-v1``); None for single_frame.
    # A template artifact applied with another aggregator would calibrate a different score.
    aggregation: str | None = None
    features: list[str] = Field(..., min_length=1)
    coefficients: list[float] | None = None
    intercept: float | None = None
    x_knots: list[float] | None = None
    y_knots: list[float] | None = None
    covered_identities: list[str] = Field(default_factory=list)
    # Gallery representation whose score distribution this artifact was fitted on. Existing schema 2
    # artifacts without this field load as "centroid".
    gallery_representation: GalleryRepresentation = "centroid"
    # Per-identity gallery revision at fit time. When an identity's live revision differs (after a
    # promotion, re-enrollment, or prototype removal), the runtime fails closed for that identity.
    gallery_revisions: dict[str, int] = Field(default_factory=dict)
    fit_basis: Literal["household_only", "pooled"]
    # The tracker crop settings every household capture shared (padding fractions, JPEG quality).
    capture_profile: dict[str, float] = Field(default_factory=dict)
    household_holdout_metrics: CalibrationMetrics = Field(default_factory=CalibrationMetrics)
    open_set_fmr_at_operating_point: float | None = None
    external_datasets: list[ExternalDatasetRef] = Field(default_factory=list)
    # The acceptance gates the fit ran under, including the template caps (records per episode,
    # coverage counted in holdout episodes), so an artifact states what it was held to.
    gates: dict[str, float] = Field(default_factory=dict)
