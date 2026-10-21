"""Runtime ArcFace calibration evaluator.

Imports only stdlib and pydantic models -- no scikit-learn. The scikit-learn toolchain lives in
app/calibration/fit.py behind the ``calibration-tools`` optional dependency group.

Health states:
  ready                 -- artifact loaded and validated; predict() can return a probability.
  degraded_missing      -- artifact_path absent or file not found.
  degraded_incompatible -- artifact versions, quality scorer, or profile do not match the service.
  degraded_invalid      -- artifact fails schema or monotonicity validation.

In any degraded state, predict() returns None. Callers must propagate None as
calibrated_confidence and must not substitute raw similarity.

Two artifact schemas load. Schema 1 maps raw similarity alone. Schema 2 maps a feature vector
(similarity plus face-quality features) and covers only the identities listed in the artifact:
any other identity, including one enrolled after the fit, gets None.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path

from app.calibration.models import FEATURE_NAMES, CalibrationArtifact, CalibrationArtifactV2

logger = logging.getLogger(__name__)

_VALID_STATES = frozenset(
    {"ready", "degraded_missing", "degraded_incompatible", "degraded_invalid"}
)
SINGLE_FRAME_PROFILE = "single_frame"

Artifact = CalibrationArtifact | CalibrationArtifactV2


def parse_artifact(raw: object) -> Artifact:
    """Validate artifact JSON as schema 2 when it says so, else as schema 1."""
    version = raw.get("schema_version", 1) if isinstance(raw, dict) else None
    if version == 2:
        return CalibrationArtifactV2.model_validate(raw)
    if version == 1:
        return CalibrationArtifact.model_validate(raw)
    raise ValueError(f"unsupported calibration artifact schema_version {version!r}")


class CalibrationEvaluator:
    """Stateless runtime evaluator for a loaded calibration artifact."""

    def __init__(
        self,
        artifact: Artifact | None,
        health_state: str,
        reason: str = "",
    ) -> None:
        if health_state not in _VALID_STATES:
            raise ValueError(f"Unknown calibration health state: {health_state!r}")
        self._artifact = artifact
        self._health_state = health_state
        self._reason = reason

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_artifact_path(
        cls,
        artifact_path: str | Path | None,
        *,
        expected_arcface_model_version: str = "",
        expected_model_profile: str = "",
        expected_preprocessing_version: str = "",
        expected_quality_version: str = "",
    ) -> CalibrationEvaluator:
        """Load a calibration artifact from disk, returning a degraded evaluator on any failure.

        ``expected_quality_version`` is the active face-quality scorer's version; a schema 2
        artifact fitted on another scorer's scores is incompatible.
        """
        if not artifact_path:
            logger.warning("calibration_artifact_missing reason=no_path_configured")
            return cls(None, "degraded_missing", "no artifact_path configured")

        p = Path(artifact_path)
        if not p.exists():
            logger.warning("calibration_artifact_missing path=%s", p)
            return cls(None, "degraded_missing", f"file not found: {p}")

        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("calibration_artifact_unreadable path=%s error=%s", p, exc)
            return cls(None, "degraded_invalid", f"read error: {exc}")

        try:
            artifact = parse_artifact(raw)
        except Exception as exc:
            logger.warning("calibration_artifact_schema_invalid path=%s error=%s", p, exc)
            return cls(None, "degraded_invalid", f"schema validation failed: {exc}")

        mismatch = _first_mismatch(
            artifact,
            arcface_model_version=expected_arcface_model_version,
            model_profile=expected_model_profile,
            preprocessing_version=expected_preprocessing_version,
            quality_version=expected_quality_version,
        )
        if mismatch is not None:
            logger.warning("calibration_artifact_incompatible field=%s", mismatch)
            return cls(None, "degraded_incompatible", f"{mismatch} mismatch")

        if not _validate_artifact(artifact):
            logger.warning(
                "calibration_artifact_invalid path=%s reason=monotonicity_or_range_check_failed",
                p,
            )
            return cls(None, "degraded_invalid", "monotonicity/range check failed")

        logger.info(
            "calibration_artifact_ready path=%s version=%s method=%s",
            p,
            artifact.artifact_version,
            artifact.method,
        )
        return cls(artifact, "ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, features: Mapping[str, float], identity_id: str | None) -> float | None:
        """Calibrated probability that a face belongs to ``identity_id``.

        ``features`` maps feature names (``similarity`` and, for schema 2, face-quality features)
        to values. Returns None in any degraded state, when a feature the artifact needs is
        missing, when a schema 2 artifact does not cover ``identity_id``, or on evaluation error.
        Callers must not substitute raw similarity when the return is None.
        """
        artifact = self._artifact
        if self._health_state != "ready" or artifact is None:
            return None
        if isinstance(artifact, CalibrationArtifactV2) and (
            identity_id is None or identity_id not in artifact.covered_identities
        ):
            return None
        try:
            return _apply(artifact, features)
        except KeyError:
            return None
        except Exception as exc:
            logger.error("calibration_predict_error error=%s", exc)
            return None

    def health(self) -> str:
        """Return the current health state string."""
        return self._health_state

    def artifact_version(self) -> str | None:
        return self._artifact.artifact_version if self._artifact else None

    def arcface_model_version(self) -> str:
        return self._artifact.arcface_model_version if self._artifact else ""

    def model_profile(self) -> str:
        return self._artifact.model_profile if self._artifact else ""

    def preprocessing_version(self) -> str:
        return self._artifact.preprocessing_version if self._artifact else ""

    def calibration_profile(self) -> str:
        """``single_frame`` for a loaded artifact (schema 1 is single-frame), else ``""``."""
        if self._artifact is None:
            return ""
        if isinstance(self._artifact, CalibrationArtifactV2):
            return self._artifact.profile
        return SINGLE_FRAME_PROFILE

    def quality_version(self) -> str:
        if isinstance(self._artifact, CalibrationArtifactV2):
            return self._artifact.quality_version
        return ""

    def covered_identities(self) -> list[str] | None:
        """Identities a schema 2 artifact covers; None for schema 1 (no per-identity coverage)."""
        if isinstance(self._artifact, CalibrationArtifactV2):
            return list(self._artifact.covered_identities)
        return None


# ------------------------------------------------------------------
# Internal helpers (no scikit-learn)
# ------------------------------------------------------------------


def _first_mismatch(
    artifact: Artifact,
    *,
    arcface_model_version: str,
    model_profile: str,
    preprocessing_version: str,
    quality_version: str,
) -> str | None:
    if arcface_model_version and artifact.arcface_model_version != arcface_model_version:
        return "arcface_model_version"
    if model_profile and artifact.model_profile != model_profile:
        return "model_profile"
    if preprocessing_version and artifact.preprocessing_version != preprocessing_version:
        return "preprocessing_version"
    if isinstance(artifact, CalibrationArtifactV2):
        if artifact.profile != SINGLE_FRAME_PROFILE:
            return "profile"
        if quality_version and artifact.quality_version != quality_version:
            return "quality_version"
    return None


def _apply(artifact: Artifact, features: Mapping[str, float]) -> float:
    """Evaluate the artifact; raises KeyError when a needed feature is missing."""
    if isinstance(artifact, CalibrationArtifactV2):
        if artifact.method == "logistic_v2":
            assert artifact.coefficients is not None
            assert artifact.intercept is not None
            z = artifact.intercept + sum(
                coef * float(features[name])
                for coef, name in zip(artifact.coefficients, artifact.features, strict=True)
            )
            return _sigmoid(z)
        assert artifact.x_knots is not None
        assert artifact.y_knots is not None
        return _isotonic_predict(artifact.x_knots, artifact.y_knots, float(features["similarity"]))
    similarity = float(features["similarity"])
    if artifact.method == "logistic":
        assert artifact.coef is not None
        assert artifact.intercept is not None
        return _sigmoid(artifact.coef * similarity + artifact.intercept)
    if artifact.method == "isotonic":
        assert artifact.x_knots is not None
        assert artifact.y_knots is not None
        return _isotonic_predict(artifact.x_knots, artifact.y_knots, similarity)
    raise ValueError(f"Unknown calibration method: {artifact.method!r}")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _isotonic_predict(x_knots: list[float], y_knots: list[float], x: float) -> float:
    """Clamped linear interpolation over isotonic regression knots (stdlib only)."""
    if x <= x_knots[0]:
        return y_knots[0]
    if x >= x_knots[-1]:
        return y_knots[-1]
    lo, hi = 0, len(x_knots) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if x_knots[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = x_knots[lo], x_knots[hi]
    y0, y1 = y_knots[lo], y_knots[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _valid_knots(x_knots: list[float] | None, y_knots: list[float] | None) -> bool:
    if x_knots is None or y_knots is None:
        return False
    if len(x_knots) != len(y_knots) or len(x_knots) < 2:
        return False
    if not all(0.0 <= y <= 1.0 and math.isfinite(y) for y in y_knots):
        return False
    return all(
        x_knots[i] >= x_knots[i - 1] and y_knots[i] >= y_knots[i - 1]
        for i in range(1, len(x_knots))
    )


def _validate_artifact(artifact: Artifact) -> bool:
    """Return True iff the artifact produces a finite mapping in [0, 1] increasing in similarity."""
    try:
        if isinstance(artifact, CalibrationArtifactV2):
            if any(name not in FEATURE_NAMES for name in artifact.features):
                return False
            if len(set(artifact.features)) != len(artifact.features):
                return False
            if "similarity" not in artifact.features:
                return False
            if artifact.method == "isotonic":
                return artifact.features == ["similarity"] and _valid_knots(
                    artifact.x_knots, artifact.y_knots
                )
            coefficients = artifact.coefficients
            if coefficients is None or artifact.intercept is None:
                return False
            if len(coefficients) != len(artifact.features):
                return False
            if not all(math.isfinite(c) for c in [*coefficients, artifact.intercept]):
                return False
            # A higher similarity must never lower the probability of a match.
            return coefficients[artifact.features.index("similarity")] > 0.0

        if artifact.method == "logistic":
            if artifact.coef is None or artifact.intercept is None:
                return False
            if not (math.isfinite(artifact.coef) and math.isfinite(artifact.intercept)):
                return False
            return all(0.0 <= _apply(artifact, {"similarity": x}) <= 1.0 for x in (-1.0, 0.0, 1.0))

        if artifact.method == "isotonic":
            return _valid_knots(artifact.x_knots, artifact.y_knots)
    except Exception:
        logger.exception("calibration_artifact_validation_error")
    return False
