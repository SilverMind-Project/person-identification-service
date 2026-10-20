"""Face image quality: how much a single detected face can be trusted for recognition.

Quality is computed here, and only here, because it needs the aligned ArcFace crop and the SCRFD
keypoints, which never leave this service. Consumers receive the score, its components, and a
version string; a calibration artifact or downstream policy that depends on the score binds to
that version.

The heuristic scorer combines four components, each mapped to [0, 1] by a clamped linear ramp,
with a weighted geometric mean. A geometric mean is used because a face that fails badly on any
one axis (an 8 px face, a heavy blur) is unusable however good the others are. Yaw is deliberately
absent: tracking-orchestrator already applies a yaw frontality factor, and counting it here too
would penalize the same turn twice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

QUALITY_VERSION_HEURISTIC_V1 = "heuristic-v1"


@dataclass(frozen=True)
class FaceQualityInputs:
    """Raw measurements of one face, in the units the components are defined over."""

    # Distance between the two SCRFD eye keypoints in pixels of the submitted image; None when the
    # keypoints are missing or degenerate.
    interocular_px: float | None
    det_score: float
    pitch_deg: float
    # The 112x112 ArcFace-aligned crop, grayscale: exactly the pixels the recognizer sees.
    aligned_gray_112: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class FaceQuality:
    """A face's quality score and the component scores it was combined from, all in [0, 1]."""

    score: float
    resolution: float
    sharpness: float
    detection: float
    pitch: float
    version: str


class FaceQualityScorer(Protocol):
    """Scores one face. Implementations are pure and deterministic for identical inputs."""

    @property
    def version(self) -> str: ...

    def score(self, inputs: FaceQualityInputs) -> FaceQuality: ...


@dataclass(frozen=True)
class HeuristicQualityParams:
    """Ramp anchors and weights for :class:`HeuristicFaceQualityScorer`.

    Each ramp maps its low anchor to 0 and its high anchor to 1. The pitch ramp is descending:
    a steeper pitch (larger absolute angle) is worse.
    """

    # Interocular distance, pixels. Recognition collapses around 10 to 12 px; ArcFace's 112x112
    # template places the eyes about 35 px apart, so larger faces are downsampled and gain little.
    iod_low_px: float = 10.0
    iod_high_px: float = 36.0
    # SCRFD detection score. The service's detection threshold is 0.6; above 0.9 the face is
    # typically clean and unoccluded.
    det_low: float = 0.5
    det_high: float = 0.9
    # log10 of the variance of the Laplacian of the grayscale aligned crop. Provisional anchors,
    # recalibrated by scripts/calibrate_face_quality.py.
    log_lap_var_low: float = math.log10(20.0)
    log_lap_var_high: float = math.log10(300.0)
    # Absolute head pitch, degrees: 15 or less is full score, 45 or more is zero.
    pitch_low_deg: float = 45.0
    pitch_high_deg: float = 15.0
    # Geometric-mean exponents; they must sum to 1.
    weight_resolution: float = 0.45
    weight_sharpness: float = 0.25
    weight_detection: float = 0.15
    weight_pitch: float = 0.15
    # Every component is floored here before exponentiation so one zero does not erase the
    # information in the others.
    component_floor: float = 0.01


def _ramp(value: float, low: float, high: float) -> float:
    """Clamped linear ramp: ``low`` maps to 0, ``high`` maps to 1 (``high < low`` descends)."""
    position = (value - low) / (high - low)
    return min(1.0, max(0.0, position))


def laplacian_variance(gray: npt.NDArray[np.uint8]) -> float:
    """Variance of the Laplacian of a grayscale image: the standard focus (sharpness) measure."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class HeuristicFaceQualityScorer:
    """Weighted geometric mean of resolution, sharpness, detection, and pitch components."""

    def __init__(self, params: HeuristicQualityParams) -> None:
        weights = (
            params.weight_resolution,
            params.weight_sharpness,
            params.weight_detection,
            params.weight_pitch,
        )
        if any(w < 0.0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError(
                f"face quality weights must be non-negative and sum to 1, got {weights}"
            )
        if not 0.0 < params.component_floor < 1.0:
            raise ValueError("face quality component_floor must be in (0, 1)")
        for low, high, name in (
            (params.iod_low_px, params.iod_high_px, "iod"),
            (params.det_low, params.det_high, "det"),
            (params.log_lap_var_low, params.log_lap_var_high, "log_lap_var"),
            (params.pitch_low_deg, params.pitch_high_deg, "pitch"),
        ):
            if low == high:
                raise ValueError(f"face quality {name} ramp anchors must differ")
        self._params = params

    @property
    def version(self) -> str:
        return QUALITY_VERSION_HEURISTIC_V1

    def score(self, inputs: FaceQualityInputs) -> FaceQuality:
        p = self._params
        # A face whose eyes could not be located fails low, never high.
        resolution = (
            0.0
            if inputs.interocular_px is None
            else _ramp(inputs.interocular_px, p.iod_low_px, p.iod_high_px)
        )
        lap_var = laplacian_variance(inputs.aligned_gray_112)
        sharpness = _ramp(math.log10(max(lap_var, 1e-6)), p.log_lap_var_low, p.log_lap_var_high)
        detection = _ramp(inputs.det_score, p.det_low, p.det_high)
        pitch = _ramp(abs(inputs.pitch_deg), p.pitch_low_deg, p.pitch_high_deg)

        floor = p.component_floor
        combined = (
            max(resolution, floor) ** p.weight_resolution
            * max(sharpness, floor) ** p.weight_sharpness
            * max(detection, floor) ** p.weight_detection
            * max(pitch, floor) ** p.weight_pitch
        )
        return FaceQuality(
            score=combined,
            resolution=resolution,
            sharpness=sharpness,
            detection=detection,
            pitch=pitch,
            version=self.version,
        )


def build_face_quality_scorer(section: Mapping[str, Any]) -> FaceQualityScorer:
    """Build the scorer named by the ``face_quality`` settings section; unknown names fail startup."""
    name = section.get("scorer")
    if name == QUALITY_VERSION_HEURISTIC_V1:
        raw = dict(section.get("heuristic") or {})
        known = {f.name for f in fields(HeuristicQualityParams)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown face_quality.heuristic keys: {sorted(unknown)}")
        params = HeuristicQualityParams(**{key: float(value) for key, value in raw.items()})
        return HeuristicFaceQualityScorer(params)
    raise ValueError(
        f"Unknown face_quality.scorer {name!r}; expected {QUALITY_VERSION_HEURISTIC_V1!r}"
    )
