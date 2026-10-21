"""CalibrationEvaluator with schema 2 artifacts (and schema 1 still loading)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from app.calibration.evaluator import CalibrationEvaluator
from app.calibration.models import CalibrationArtifactV2

_V2: dict[str, Any] = {
    "schema_version": 2,
    "artifact_version": "a-v2",
    "method": "logistic_v2",
    "arcface_model_version": "buffalo_l",
    "model_profile": "full",
    "preprocessing_version": "v1",
    "fitted_at": "2026-10-21T09:00:00+00:00",
    "profile": "single_frame",
    "quality_version": "heuristic-v1",
    "features": ["similarity", "quality_score"],
    "coefficients": [12.0, 3.0],
    "intercept": -7.0,
    "covered_identities": ["resident-alpha"],
    "fit_basis": "household_only",
}
_V1: dict[str, Any] = {
    "schema_version": 1,
    "artifact_version": "a-v1",
    "method": "logistic",
    "arcface_model_version": "buffalo_l",
    "model_profile": "full",
    "preprocessing_version": "v1",
    "fitted_at": "2026-10-21T09:00:00+00:00",
    "coef": 10.0,
    "intercept": -5.0,
}
_FEATURES = {"similarity": 0.5, "quality_score": 0.4}


def _load(tmp_path: Path, artifact: dict[str, Any], **expected: str) -> CalibrationEvaluator:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact))
    return CalibrationEvaluator.from_artifact_path(
        path,
        expected_arcface_model_version=expected.get("arcface", "buffalo_l"),
        expected_model_profile="full",
        expected_preprocessing_version="v1",
        expected_quality_version=expected.get("quality", "heuristic-v1"),
    )


def test_v2_logistic_applies_raw_space_coefficients(tmp_path: Path) -> None:
    ev = _load(tmp_path, _V2)
    assert ev.health() == "ready"
    expected = 1.0 / (1.0 + math.exp(-(-7.0 + 12.0 * 0.5 + 3.0 * 0.4)))
    assert ev.predict(_FEATURES, "resident-alpha") == pytest.approx(expected)
    assert (ev.calibration_profile(), ev.quality_version()) == ("single_frame", "heuristic-v1")
    assert ev.covered_identities() == ["resident-alpha"]


def test_v2_fails_closed_for_identities_it_does_not_cover(tmp_path: Path) -> None:
    ev = _load(tmp_path, _V2)
    assert ev.predict(_FEATURES, "resident-beta") is None
    assert ev.predict(_FEATURES, None) is None


def test_v2_missing_feature_gives_none(tmp_path: Path) -> None:
    assert _load(tmp_path, _V2).predict({"similarity": 0.5}, "resident-alpha") is None


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"quality": "heuristic-v2"}, "degraded_incompatible"),
        ({"arcface": "other"}, "degraded_incompatible"),
    ],
)
def test_version_mismatch_is_incompatible(
    tmp_path: Path, changes: dict[str, str], expected: str
) -> None:
    assert _load(tmp_path, _V2, **changes).health() == expected


def test_template_profile_is_incompatible_for_single_frame(tmp_path: Path) -> None:
    ev = _load(tmp_path, _V2 | {"profile": "template"})
    assert ev.health() == "degraded_incompatible"
    assert ev.predict(_FEATURES, "resident-alpha") is None


@pytest.mark.parametrize(
    "changes",
    [
        {"coefficients": [-12.0, 3.0]},  # higher similarity would lower the match probability
        {"coefficients": [12.0]},  # length mismatch
        {"features": ["similarity", "colour"]},
        {"features": ["quality_score", "detection"]},  # no similarity
        {"method": "isotonic", "x_knots": [0.1, 0.5], "y_knots": [0.0, 0.9]},  # two features
    ],
)
def test_invalid_v2_artifacts_are_rejected(tmp_path: Path, changes: dict[str, Any]) -> None:
    assert _load(tmp_path, _V2 | changes).health() == "degraded_invalid"


def test_v2_isotonic_on_similarity(tmp_path: Path) -> None:
    artifact = _V2 | {
        "method": "isotonic",
        "features": ["similarity"],
        "coefficients": None,
        "intercept": None,
        "x_knots": [0.2, 0.6],
        "y_knots": [0.0, 1.0],
    }
    ev = _load(tmp_path, artifact)
    assert ev.predict({"similarity": 0.4}, "resident-alpha") == pytest.approx(0.5)


def test_v1_still_loads_and_ignores_identity_and_quality(tmp_path: Path) -> None:
    ev = _load(tmp_path, _V1)
    assert ev.health() == "ready"
    expected = 1.0 / (1.0 + math.exp(-(10.0 * 0.5 - 5.0)))
    assert ev.predict({"similarity": 0.5}, "anyone") == pytest.approx(expected)
    assert (ev.calibration_profile(), ev.quality_version(), ev.covered_identities()) == (
        "single_frame",
        "",
        None,
    )


@pytest.mark.parametrize("artifact", [_V2 | {"schema_version": 3}, _V1 | {"schema_version": 3}])
def test_unknown_schema_version_is_invalid(tmp_path: Path, artifact: dict[str, Any]) -> None:
    assert _load(tmp_path, artifact).health() == "degraded_invalid"


def test_published_v2_json_schema_matches_the_model() -> None:
    published = json.loads(
        (Path(__file__).parent.parent / "schemas/calibration_artifact_v2.schema.json").read_text()
    )
    expected = CalibrationArtifactV2.model_json_schema()
    expected["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    expected["$id"] = "calibration_artifact_v2.schema.json"
    assert published == expected, "regenerate schemas/calibration_artifact_v2.schema.json"
