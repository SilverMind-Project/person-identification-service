"""Recognizer profile configuration and validation.

A recognizer profile bundles the Triton model names, alignment template, input dimensions,
pixel normalization, embedding dimensionality, preprocessing version, and model licence terms
as one immutable unit. Swapping recognizers changes the active profile name in settings rather
than editing matching policy, calibration code, or downstream services.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

DEFAULT_RECOGNIZER_VERSION = "buffalo_l"
DEFAULT_PROFILE_NAME = "buffalo_l"
DEFAULT_EMBEDDING_DIM = 512

ColorSpace = Literal["rgb", "bgr"]

# Standard five-point 112x112 InsightFace / ArcFace alignment destinations (left eye, right eye,
# nose tip, left mouth corner, right mouth corner).
DEFAULT_ARCFACE_TEMPLATE: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)

BUFFALO_L_LICENCE = (
    "InsightFace model zoo: non-commercial research purposes only "
    "(code MIT; weights and training data non-commercial research only)"
)

ADAFACE_WEBFACE4M_LICENCE = (
    "AdaFace code MIT; pretrained weights trained on WebFace4M "
    "(non-commercial academic research only; training-data licence inherits into weights)"
)


@dataclass(frozen=True)
class RecognizerProfile:
    """Immutable specification of a Triton-backed face recognition profile.

    Keeping ``landmark_3d_model`` separate from ``recognition_model`` allows swapping the 112x112
    recognition head without altering the 68-point 3D landmark head-pose estimator.
    """

    name: str
    recognizer_version: str
    detector_model: str
    recognition_model: str
    landmark_2d_model: str
    landmark_3d_model: str
    genderage_model: str
    input_size_px: int = 112
    alignment_template: tuple[tuple[float, float], ...] = DEFAULT_ARCFACE_TEMPLATE
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    preprocessing_version: str = "v1"
    licence: str = BUFFALO_L_LICENCE
    normalization_mean: float = 127.5
    normalization_std: float = 127.5
    color_space: ColorSpace = "rgb"
    recognition_input_name: str = "input.1"
    recognition_output_name: str = "683"
    # Optional explicit quality-version scope. When None, the baseline buffalo_l profile keeps
    # bare scorer versions (for example "heuristic-v1") while every other profile qualifies
    # quality scores as "<base>@<recognizer_version>" so quality consumers fail closed across
    # recognizer switches.
    quality_version_scope: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "recognizer_version",
            "detector_model",
            "recognition_model",
            "landmark_2d_model",
            "landmark_3d_model",
            "genderage_model",
            "preprocessing_version",
            "licence",
            "recognition_input_name",
            "recognition_output_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RecognizerProfile.{field_name} must be a non-empty string")

        models = (
            self.detector_model,
            self.recognition_model,
            self.landmark_2d_model,
            self.landmark_3d_model,
            self.genderage_model,
        )
        if len(set(models)) != len(models):
            raise ValueError("RecognizerProfile model names must be distinct")

        if self.input_size_px <= 0:
            raise ValueError(
                f"RecognizerProfile.input_size_px must be > 0, got {self.input_size_px}"
            )
        if self.embedding_dim <= 0:
            raise ValueError(
                f"RecognizerProfile.embedding_dim must be > 0, got {self.embedding_dim}"
            )
        if not math.isfinite(self.normalization_mean):
            raise ValueError("RecognizerProfile.normalization_mean must be finite")
        if not math.isfinite(self.normalization_std) or self.normalization_std <= 0.0:
            raise ValueError("RecognizerProfile.normalization_std must be finite and > 0")
        if self.color_space not in ("rgb", "bgr"):
            raise ValueError(
                f"RecognizerProfile.color_space must be 'rgb' or 'bgr', got {self.color_space!r}"
            )
        if len(self.alignment_template) != 5 or any(
            len(point) != 2 or not (math.isfinite(point[0]) and math.isfinite(point[1]))
            for point in self.alignment_template
        ):
            raise ValueError(
                "RecognizerProfile.alignment_template must have shape (5, 2) with finite coordinates"
            )

    @property
    def model_names(self) -> dict[str, str]:
        """Triton model map keyed by pipeline role."""
        return {
            "detector": self.detector_model,
            "recognition": self.recognition_model,
            "landmark_2d": self.landmark_2d_model,
            "landmark_3d": self.landmark_3d_model,
            "genderage": self.genderage_model,
        }

    def alignment_template_array(self) -> npt.NDArray[np.float32]:
        """Return a fresh (5, 2) float32 destination array for similarity alignment."""
        return np.array(self.alignment_template, dtype=np.float32)

    def scoped_quality_version(self, base_scorer_version: str) -> str:
        """Bind a face-quality scorer version to this recognizer profile.

        The baseline buffalo_l profile preserves the bare scorer version (for example
        ``heuristic-v1``) for backward compatibility with existing captures and CTS settings. Any
        other profile qualifies the scorer version with ``@<recognizer_version>`` so downstream
        quality consumers (calibration artifacts, prototypes, attestation, CTS face quality)
        automatically reject scores from a different recognizer.
        """
        if self.quality_version_scope is not None:
            scope = self.quality_version_scope.strip()
            return f"{base_scorer_version}@{scope}" if scope else base_scorer_version
        if self.recognizer_version == DEFAULT_RECOGNIZER_VERSION:
            return base_scorer_version
        return f"{base_scorer_version}@{self.recognizer_version}"

    def templates_compatible(self, other: RecognizerProfile, *, atol: float = 1e-4) -> bool:
        """True when aligned crops produced by *other* match this profile's input size and warp."""
        if self.input_size_px != other.input_size_px:
            return False
        return bool(
            np.allclose(
                self.alignment_template_array(),
                other.alignment_template_array(),
                atol=atol,
            )
        )


BUFFALO_L_PROFILE = RecognizerProfile(
    name=DEFAULT_PROFILE_NAME,
    recognizer_version=DEFAULT_RECOGNIZER_VERSION,
    detector_model="face-detector-scrfd",
    recognition_model="face-recognition-arcface",
    landmark_2d_model="face-landmark-2d106",
    landmark_3d_model="face-landmark-3d68",
    genderage_model="face-attribute-genderage",
    input_size_px=112,
    alignment_template=DEFAULT_ARCFACE_TEMPLATE,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    preprocessing_version="v1",
    licence=BUFFALO_L_LICENCE,
    normalization_mean=127.5,
    normalization_std=127.5,
    color_space="rgb",
    recognition_input_name="input.1",
    recognition_output_name="683",
)


def _parse_template(raw: Any) -> tuple[tuple[float, float], ...]:
    if raw is None:
        return DEFAULT_ARCFACE_TEMPLATE
    if not isinstance(raw, list | tuple) or len(raw) != 5:
        raise ValueError("alignment_template must be a list of 5 [x, y] coordinate pairs")
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError("each alignment_template entry must be a [x, y] pair")
        points.append((float(item[0]), float(item[1])))
    return tuple(points)


def _profile_from_mapping(name: str, raw: Mapping[str, Any]) -> RecognizerProfile:
    if not isinstance(raw, Mapping):
        raise ValueError(f"face_engine.profiles.{name} must be a mapping")
    color_raw = str(raw.get("color_space", "rgb")).strip().lower()
    if color_raw not in ("rgb", "bgr"):
        raise ValueError(
            f"face_engine.profiles.{name}.color_space must be 'rgb' or 'bgr', got {color_raw!r}"
        )
    scope_raw = raw.get("quality_version_scope")
    return RecognizerProfile(
        name=name,
        recognizer_version=str(raw.get("recognizer_version", name)),
        detector_model=str(raw.get("detector_model", "face-detector-scrfd")),
        recognition_model=str(raw.get("recognition_model", "face-recognition-arcface")),
        landmark_2d_model=str(raw.get("landmark_2d_model", "face-landmark-2d106")),
        landmark_3d_model=str(raw.get("landmark_3d_model", "face-landmark-3d68")),
        genderage_model=str(raw.get("genderage_model", "face-attribute-genderage")),
        input_size_px=int(raw.get("input_size_px", 112)),
        alignment_template=_parse_template(raw.get("alignment_template")),
        embedding_dim=int(raw.get("embedding_dim", DEFAULT_EMBEDDING_DIM)),
        preprocessing_version=str(raw.get("preprocessing_version", "v1")),
        licence=str(raw.get("licence", "")),
        normalization_mean=float(raw.get("normalization_mean", 127.5)),
        normalization_std=float(raw.get("normalization_std", 127.5)),
        color_space=color_raw,  # type: ignore[arg-type]
        recognition_input_name=str(raw.get("recognition_input_name", "input.1")),
        recognition_output_name=str(raw.get("recognition_output_name", "683")),
        quality_version_scope=None if scope_raw is None else str(scope_raw),
    )


def _legacy_buffalo_profile(face_engine_cfg: Mapping[str, Any]) -> RecognizerProfile:
    models = face_engine_cfg.get("models")
    models_map: Mapping[str, Any] = models if isinstance(models, Mapping) else {}
    return RecognizerProfile(
        name=DEFAULT_PROFILE_NAME,
        recognizer_version=str(face_engine_cfg.get("model_name", DEFAULT_RECOGNIZER_VERSION)),
        detector_model=str(models_map.get("detector", BUFFALO_L_PROFILE.detector_model)),
        recognition_model=str(models_map.get("recognition", BUFFALO_L_PROFILE.recognition_model)),
        landmark_2d_model=str(models_map.get("landmark_2d", BUFFALO_L_PROFILE.landmark_2d_model)),
        landmark_3d_model=str(models_map.get("landmark_3d", BUFFALO_L_PROFILE.landmark_3d_model)),
        genderage_model=str(models_map.get("genderage", BUFFALO_L_PROFILE.genderage_model)),
        input_size_px=BUFFALO_L_PROFILE.input_size_px,
        alignment_template=BUFFALO_L_PROFILE.alignment_template,
        embedding_dim=BUFFALO_L_PROFILE.embedding_dim,
        preprocessing_version=BUFFALO_L_PROFILE.preprocessing_version,
        licence=BUFFALO_L_LICENCE,
    )


def load_recognizer_profiles(
    face_engine_cfg: Mapping[str, Any] | None,
) -> dict[str, RecognizerProfile]:
    """Parse and validate every profile under ``face_engine.profiles``."""
    cfg: Mapping[str, Any] = face_engine_cfg or {}
    raw_profiles = cfg.get("profiles")
    if raw_profiles is None:
        return {DEFAULT_PROFILE_NAME: _legacy_buffalo_profile(cfg)}
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ValueError("face_engine.profiles must be a non-empty mapping")
    return {
        str(name): _profile_from_mapping(str(name), spec) for name, spec in raw_profiles.items()
    }


def get_recognizer_profile(
    face_engine_cfg: Mapping[str, Any] | None,
    profile_name: str,
) -> RecognizerProfile:
    """Return the validated profile named *profile_name*; raises ``ValueError`` if unknown."""
    profiles = load_recognizer_profiles(face_engine_cfg)
    if profile_name not in profiles:
        known = sorted(profiles)
        raise ValueError(
            f"Unknown face recognizer profile {profile_name!r}; configured profiles: {known}"
        )
    return profiles[profile_name]


def load_active_recognizer_profile(
    face_engine_cfg: Mapping[str, Any] | None,
) -> RecognizerProfile:
    """Load the profile named by ``face_engine.active_profile``; fails fast on unknown names."""
    cfg: Mapping[str, Any] = face_engine_cfg or {}
    active = str(cfg.get("active_profile", DEFAULT_PROFILE_NAME)).strip()
    if not active:
        raise ValueError("face_engine.active_profile must not be empty")
    return get_recognizer_profile(cfg, active)
