"""Runtime ArcFace calibration evaluator.

Imports only stdlib and numpy -- no scikit-learn.  The scikit-learn
toolchain lives in app/calibration/fit.py behind the ``calibration-tools``
optional dependency group.

Health states:
  ready                 -- artifact loaded and validated; predict() returns float.
  degraded_missing      -- artifact_path absent or file not found.
  degraded_incompatible -- artifact versions do not match configured expectations.
  degraded_invalid      -- artifact fails schema or monotonicity validation.

In any degraded state, predict() returns None.  Callers must propagate
None as calibrated_confidence and must not substitute raw_similarity.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from app.calibration.models import CalibrationArtifact

logger = logging.getLogger(__name__)

_VALID_STATES = frozenset(
    {"ready", "degraded_missing", "degraded_incompatible", "degraded_invalid"}
)


class CalibrationEvaluator:
    """Stateless runtime evaluator for a loaded calibration artifact."""

    def __init__(
        self,
        artifact: CalibrationArtifact | None,
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
    ) -> CalibrationEvaluator:
        """Load a calibration artifact from disk, returning a degraded evaluator on any failure."""
        if not artifact_path:
            logger.warning(
                "calibration_artifact_missing reason=no_path_configured"
            )
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
            artifact = CalibrationArtifact.model_validate(raw)
        except Exception as exc:
            logger.warning("calibration_artifact_schema_invalid path=%s error=%s", p, exc)
            return cls(None, "degraded_invalid", f"schema validation failed: {exc}")

        if expected_arcface_model_version and (
            artifact.arcface_model_version != expected_arcface_model_version
        ):
            logger.warning(
                "calibration_artifact_incompatible field=arcface_model_version "
                "artifact=%s expected=%s",
                artifact.arcface_model_version,
                expected_arcface_model_version,
            )
            return cls(None, "degraded_incompatible", "arcface_model_version mismatch")

        if expected_model_profile and artifact.model_profile != expected_model_profile:
            logger.warning(
                "calibration_artifact_incompatible field=model_profile "
                "artifact=%s expected=%s",
                artifact.model_profile,
                expected_model_profile,
            )
            return cls(None, "degraded_incompatible", "model_profile mismatch")

        if expected_preprocessing_version and (
            artifact.preprocessing_version != expected_preprocessing_version
        ):
            logger.warning(
                "calibration_artifact_incompatible field=preprocessing_version "
                "artifact=%s expected=%s",
                artifact.preprocessing_version,
                expected_preprocessing_version,
            )
            return cls(None, "degraded_incompatible", "preprocessing_version mismatch")

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

    def predict(self, raw_similarity: float) -> float | None:
        """Apply calibration to a raw cosine similarity.

        Returns None in any degraded health state or on evaluation error.
        Callers must not substitute raw_similarity when the return is None.
        """
        if self._health_state != "ready" or self._artifact is None:
            return None
        try:
            return _apply(self._artifact, raw_similarity)
        except Exception as exc:
            logger.error("calibration_predict_error similarity=%s error=%s", raw_similarity, exc)
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


# ------------------------------------------------------------------
# Internal helpers (no scikit-learn)
# ------------------------------------------------------------------


def _apply(artifact: CalibrationArtifact, x: float) -> float:
    if artifact.method == "logistic":
        assert artifact.coef is not None
        assert artifact.intercept is not None
        return 1.0 / (1.0 + math.exp(-(artifact.coef * x + artifact.intercept)))
    if artifact.method == "isotonic":
        assert artifact.x_knots is not None
        assert artifact.y_knots is not None
        return _isotonic_predict(artifact.x_knots, artifact.y_knots, x)
    raise ValueError(f"Unknown calibration method: {artifact.method!r}")


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


def _validate_artifact(artifact: CalibrationArtifact) -> bool:
    """Return True iff the artifact produces a finite monotone mapping in [0, 1]."""
    try:
        if artifact.method == "logistic":
            if artifact.coef is None or artifact.intercept is None:
                return False
            if not (math.isfinite(artifact.coef) and math.isfinite(artifact.intercept)):
                return False
            for x in (-1.0, 0.0, 1.0):
                p = _apply(artifact, x)
                if not (0.0 <= p <= 1.0 and math.isfinite(p)):
                    return False
            return True

        if artifact.method == "isotonic":
            if artifact.x_knots is None or artifact.y_knots is None:
                return False
            if len(artifact.x_knots) != len(artifact.y_knots):
                return False
            if len(artifact.x_knots) < 2:
                return False
            for y in artifact.y_knots:
                if not (0.0 <= y <= 1.0 and math.isfinite(y)):
                    return False
            for i in range(1, len(artifact.x_knots)):
                if artifact.x_knots[i] < artifact.x_knots[i - 1]:
                    return False
                if artifact.y_knots[i] < artifact.y_knots[i - 1]:
                    return False
            return True
    except Exception:
        pass
    return False
