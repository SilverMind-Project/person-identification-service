"""M10 ArcFace calibration toolchain and runtime contract tests.

Tests are sklearn-free (evaluator is stdlib/numpy only).
Fixtures under tests/fixtures/identity_integrity/ are committed JSON.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from app.calibration.evaluator import CalibrationEvaluator
from app.calibration.models import (
    CalibrationMetrics,
    SplitCounts,
)
from app.models.identification import FaceDetection

_FIXTURES = Path(__file__).parent / "fixtures/identity_integrity"
_LOGISTIC_ARTIFACT = _FIXTURES / "calibration_artifact_logistic.json"
_ISOTONIC_ARTIFACT = _FIXTURES / "calibration_artifact_isotonic.json"


# ---------------------------------------------------------------------------
# Helper: build an artifact dict
# ---------------------------------------------------------------------------


def _artifact_dict(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid logistic artifact dict with optional overrides."""
    base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_version": "test-v1",
        "method": "logistic",
        "arcface_model_version": "buffalo_l",
        "model_profile": "full",
        "preprocessing_version": "v1",
        "fitted_at": "2026-06-01T00:00:00Z",
        "coef": 5.0,
        "intercept": -2.0,
        "x_knots": None,
        "y_knots": None,
        "split_counts": {
            "fit_pairs": 400, "fit_positives": 200, "fit_negatives": 200,
            "val_pairs": 100, "val_positives": 50, "val_negatives": 50,
            "test_pairs": 100, "test_positives": 50, "test_negatives": 50,
            "fit_identities": 20, "val_identities": 5, "test_identities": 5,
        },
        "metrics": {
            "brier_score": 0.05, "log_loss": 0.15, "ece": 0.02,
            "fmr_at_operating_point": 0.01, "fnmr_at_operating_point": 0.08,
            "operating_point_threshold": 0.80,
        },
    }
    base.update(overrides)
    return base


def _make_evaluator(**overrides: Any) -> CalibrationEvaluator:
    """Build an evaluator from an artifact dict (all version checks disabled)."""
    import tempfile

    raw = _artifact_dict(**overrides)
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(raw, f)
        path = f.name
    return CalibrationEvaluator.from_artifact_path(path)


# ---------------------------------------------------------------------------
# 1. Evaluator health states
# ---------------------------------------------------------------------------


def test_evaluator_missing_path_is_degraded_missing() -> None:
    ev = CalibrationEvaluator.from_artifact_path(None)
    assert ev.health() == "degraded_missing"
    assert ev.predict(0.8) is None


def test_evaluator_nonexistent_path_is_degraded_missing(tmp_path: Path) -> None:
    ev = CalibrationEvaluator.from_artifact_path(tmp_path / "nope.json")
    assert ev.health() == "degraded_missing"
    assert ev.predict(0.8) is None


def test_evaluator_malformed_json_is_degraded_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(bad)
    assert ev.health() == "degraded_invalid"
    assert ev.predict(0.8) is None


def test_evaluator_schema_invalid_is_degraded_invalid(tmp_path: Path) -> None:
    # Missing required field "method"
    raw = _artifact_dict()
    del raw["method"]
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(p)
    assert ev.health() == "degraded_invalid"
    assert ev.predict(0.8) is None


def test_evaluator_nonfinite_coef_is_degraded_invalid(tmp_path: Path) -> None:
    raw = _artifact_dict(coef=float("inf"))
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(p)
    assert ev.health() == "degraded_invalid"
    assert ev.predict(0.8) is None


def test_evaluator_version_mismatch_is_degraded_incompatible(tmp_path: Path) -> None:
    raw = _artifact_dict(arcface_model_version="some_other_model")
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(
        p,
        expected_arcface_model_version="buffalo_l",
    )
    assert ev.health() == "degraded_incompatible"
    assert ev.predict(0.8) is None


def test_evaluator_model_profile_mismatch_is_degraded_incompatible(tmp_path: Path) -> None:
    raw = _artifact_dict(model_profile="int8")
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(
        p,
        expected_model_profile="full",
    )
    assert ev.health() == "degraded_incompatible"
    assert ev.predict(0.8) is None


def test_evaluator_preprocessing_version_mismatch_is_degraded_incompatible(tmp_path: Path) -> None:
    raw = _artifact_dict(preprocessing_version="v2")
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(
        p,
        expected_preprocessing_version="v1",
    )
    assert ev.health() == "degraded_incompatible"
    assert ev.predict(0.8) is None


def test_evaluator_valid_logistic_is_ready() -> None:
    ev = _make_evaluator()
    assert ev.health() == "ready"
    assert ev.artifact_version() == "test-v1"
    assert ev.arcface_model_version() == "buffalo_l"
    assert ev.model_profile() == "full"
    assert ev.preprocessing_version() == "v1"


def test_evaluator_valid_isotonic_is_ready(tmp_path: Path) -> None:
    raw = _artifact_dict(
        method="isotonic",
        coef=None,
        intercept=None,
        x_knots=[0.0, 0.5, 1.0],
        y_knots=[0.0, 0.5, 1.0],
    )
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(p)
    assert ev.health() == "ready"


def test_evaluator_isotonic_non_monotone_is_degraded_invalid(tmp_path: Path) -> None:
    raw = _artifact_dict(
        method="isotonic",
        coef=None,
        intercept=None,
        x_knots=[0.0, 0.5, 1.0],
        y_knots=[0.0, 0.9, 0.3],  # not monotone
    )
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    ev = CalibrationEvaluator.from_artifact_path(p)
    assert ev.health() == "degraded_invalid"
    assert ev.predict(0.5) is None


# ---------------------------------------------------------------------------
# 2. Logistic evaluator matches the expected formula (sklearn-free)
# ---------------------------------------------------------------------------

# These expected values were computed with: 1/(1+exp(-(5*x - 2)))
_LOGISTIC_CASES: list[tuple[float, float]] = [
    (0.0, 1.0 / (1.0 + math.exp(2.0))),
    (0.4, 0.5),  # coef*0.4 + intercept = 0
    (0.75, 1.0 / (1.0 + math.exp(-1.75))),
    (1.0, 1.0 / (1.0 + math.exp(-3.0))),
]


def test_logistic_evaluator_matches_formula() -> None:
    ev = _make_evaluator(coef=5.0, intercept=-2.0)
    for sim, expected in _LOGISTIC_CASES:
        got = ev.predict(sim)
        assert got is not None
        assert abs(got - expected) < 1e-9, f"sim={sim}: got {got}, expected {expected}"


def test_logistic_evaluator_output_in_range() -> None:
    ev = _make_evaluator()
    for sim in [-1.0, -0.5, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5]:
        result = ev.predict(sim)
        assert result is not None
        assert 0.0 <= result <= 1.0, f"out of range for sim={sim}: {result}"


# ---------------------------------------------------------------------------
# 3. Isotonic evaluator: clamped linear interpolation (stdlib only)
# ---------------------------------------------------------------------------

_ISO_KNOTS_X = [0.0, 0.5, 1.0]
_ISO_KNOTS_Y = [0.0, 0.5, 1.0]  # identity line for easy testing


def _make_isotonic_evaluator(
    x_knots: list[float] | None = None,
    y_knots: list[float] | None = None,
    tmp_path: Path | None = None,
) -> CalibrationEvaluator:
    import tempfile

    raw = _artifact_dict(
        method="isotonic",
        coef=None,
        intercept=None,
        x_knots=x_knots or _ISO_KNOTS_X,
        y_knots=y_knots or _ISO_KNOTS_Y,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(raw, f)
        path = f.name
    return CalibrationEvaluator.from_artifact_path(path)


def test_isotonic_evaluator_identity_line() -> None:
    ev = _make_isotonic_evaluator()
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = ev.predict(x)
        assert result is not None
        assert abs(result - x) < 1e-9, f"expected {x}, got {result}"


def test_isotonic_evaluator_clamp_below() -> None:
    ev = _make_isotonic_evaluator()
    result = ev.predict(-0.5)
    assert result is not None
    assert abs(result - 0.0) < 1e-9


def test_isotonic_evaluator_clamp_above() -> None:
    ev = _make_isotonic_evaluator()
    result = ev.predict(2.0)
    assert result is not None
    assert abs(result - 1.0) < 1e-9


def test_isotonic_evaluator_nonidentity_knots() -> None:
    # knots: 0→0, 0.5→0.3, 1.0→0.9
    ev = _make_isotonic_evaluator(
        x_knots=[0.0, 0.5, 1.0],
        y_knots=[0.0, 0.3, 0.9],
    )
    # at x=0.5, expect 0.3
    result_mid = ev.predict(0.5)
    assert result_mid is not None
    assert abs(result_mid - 0.3) < 1e-9
    # at x=0.75, expect 0.3 + (0.9-0.3)*(0.75-0.5)/(1.0-0.5) = 0.3 + 0.6*0.5 = 0.6
    result_75 = ev.predict(0.75)
    assert result_75 is not None
    assert abs(result_75 - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# 4. Committed fixture artifacts
# ---------------------------------------------------------------------------


def test_committed_logistic_artifact_is_ready() -> None:
    ev = CalibrationEvaluator.from_artifact_path(
        _LOGISTIC_ARTIFACT,
        expected_arcface_model_version="buffalo_l",
        expected_model_profile="full",
        expected_preprocessing_version="v1",
    )
    assert ev.health() == "ready"


def test_committed_isotonic_artifact_is_ready() -> None:
    ev = CalibrationEvaluator.from_artifact_path(
        _ISOTONIC_ARTIFACT,
        expected_arcface_model_version="buffalo_l",
        expected_model_profile="full",
        expected_preprocessing_version="v1",
    )
    assert ev.health() == "ready"


def test_committed_logistic_artifact_at_known_point() -> None:
    """The committed logistic artifact uses coef=5.0, intercept=-2.0."""
    ev = CalibrationEvaluator.from_artifact_path(_LOGISTIC_ARTIFACT)
    p_at_04 = ev.predict(0.4)  # at decision boundary
    assert p_at_04 is not None
    assert abs(p_at_04 - 0.5) < 1e-9


def test_committed_isotonic_artifact_monotone() -> None:
    ev = CalibrationEvaluator.from_artifact_path(_ISOTONIC_ARTIFACT)
    prev = -1.0
    for x in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p = ev.predict(x)
        assert p is not None
        assert p >= prev - 1e-9, f"non-monotone at x={x}: {p} < {prev}"
        prev = p


# ---------------------------------------------------------------------------
# 5. FaceDetection response model: raw/calibrated/det separation
# ---------------------------------------------------------------------------


def test_face_detection_separates_raw_and_calibrated() -> None:
    det = FaceDetection(
        person_id="resident-alpha",
        name="Alice",
        confidence=0.83,
        bbox=[10.0, 20.0, 110.0, 220.0],
        recognition_state="recognized",
        similarity=0.83,
        calibrated_confidence=0.72,
        calibration_status="ready",
        calibration_artifact_version="test-v1",
        arcface_model_version="buffalo_l",
        model_profile="full",
        preprocessing_version="v1",
    ).model_dump()

    assert det["raw_similarity"] == 0.83
    assert det["similarity"] == 0.83
    assert det["calibrated_confidence"] == 0.72
    assert det["calibration_status"] == "ready"
    assert det["calibration_artifact_version"] == "test-v1"
    assert det["arcface_model_version"] == "buffalo_l"
    # raw and calibrated are always distinct fields
    assert det["raw_similarity"] != det["calibrated_confidence"]


def test_face_detection_calibrated_none_when_degraded() -> None:
    det = FaceDetection(
        person_id="unknown",
        name="Unknown",
        confidence=0.75,
        bbox=[0.0, 0.0, 100.0, 200.0],
        recognition_state="recognized",
        similarity=0.75,
    ).model_dump()

    assert det["calibrated_confidence"] is None
    assert det["calibration_status"] == "degraded_missing"
    assert det["raw_similarity"] == 0.75


def test_face_detection_raw_recognition_state_unaffected_by_calibration() -> None:
    """Recognition state must not change based on calibration availability."""
    base_kwargs = {
        "person_id": "resident-beta",
        "name": "Bob",
        "confidence": 0.76,
        "bbox": [0.0, 0.0, 50.0, 100.0],
        "recognition_state": "recognized",
        "similarity": 0.76,
    }
    with_cal = FaceDetection(**base_kwargs, calibrated_confidence=0.9, calibration_status="ready").model_dump()
    without_cal = FaceDetection(**base_kwargs).model_dump()

    assert with_cal["recognition_state"] == "recognized"
    assert without_cal["recognition_state"] == "recognized"
    assert with_cal["recognition_state"] == without_cal["recognition_state"]


def test_face_detection_candidate_state_has_null_calibrated() -> None:
    det = FaceDetection(
        person_id="person-001",
        name="Alice",
        confidence=0.30,
        bbox=[0.0, 0.0, 50.0, 100.0],
        recognition_state="candidate",
        similarity=0.30,
    ).model_dump()
    # Candidate anchors should not carry calibrated authority
    assert det["calibrated_confidence"] is None


def test_face_detection_unrecognized_state_has_null_calibrated() -> None:
    det = FaceDetection(
        person_id="unknown",
        name="Unknown",
        confidence=0.10,
        bbox=[0.0, 0.0, 50.0, 100.0],
        recognition_state="unrecognized",
        similarity=0.10,
    ).model_dump()
    assert det["calibrated_confidence"] is None


# ---------------------------------------------------------------------------
# 6. Sklearn absent: evaluator imports cleanly
# ---------------------------------------------------------------------------


def test_evaluator_imports_without_sklearn() -> None:
    """CalibrationEvaluator must not require scikit-learn at import or predict time."""
    with patch.dict(sys.modules, {"sklearn": None, "sklearn.linear_model": None,
                                   "sklearn.isotonic": None}):
        # Re-import to verify it loads cleanly with sklearn blocked
        import importlib

        import app.calibration.evaluator as ev_mod
        importlib.reload(ev_mod)

        ev = ev_mod.CalibrationEvaluator.from_artifact_path(None)
        assert ev.health() == "degraded_missing"


# ---------------------------------------------------------------------------
# 7. Calibration fit.py (sklearn required; skip if absent)
# ---------------------------------------------------------------------------

sklearn_available = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("sklearn"),
    reason="calibration-tools not installed",
)


@sklearn_available
def test_fit_logistic_deterministic() -> None:
    from app.calibration.fit import fit_logistic

    pairs = [(float(i) / 100, i % 2) for i in range(200)]
    coef1, intercept1 = fit_logistic(pairs, random_seed=42)
    coef2, intercept2 = fit_logistic(pairs, random_seed=42)
    assert coef1 == coef2
    assert intercept1 == intercept2


@sklearn_available
def test_fit_isotonic_produces_monotone_knots() -> None:
    from app.calibration.fit import fit_isotonic

    # positives cluster high, negatives cluster low
    pairs = [(0.3 + 0.01 * i, 0) for i in range(50)] + [
        (0.7 + 0.01 * i, 1) for i in range(50)
    ]
    x_knots, y_knots = fit_isotonic(pairs)
    assert len(x_knots) == len(y_knots)
    for i in range(1, len(x_knots)):
        assert x_knots[i] >= x_knots[i - 1], "x_knots must be non-decreasing"
        assert y_knots[i] >= y_knots[i - 1], "y_knots must be non-decreasing"
    for y in y_knots:
        assert 0.0 <= y <= 1.0, f"y_knots out of [0,1]: {y}"


# ---------------------------------------------------------------------------
# 8. Leakage rejection
# ---------------------------------------------------------------------------


def test_leakage_rejection_identity_overlap() -> None:
    from app.calibration.fit import validate_split_disjoint

    fit_ids = {"id-1", "id-2", "id-3"}
    val_ids = {"id-3", "id-4"}  # id-3 leaks
    test_ids = {"id-5"}
    findings = validate_split_disjoint(fit_ids, val_ids, test_ids)
    assert findings, "Expected leakage to be detected"
    assert any("id-3" in f for f in findings)


def test_leakage_rejection_clean_split() -> None:
    from app.calibration.fit import validate_split_disjoint

    fit_ids = {"id-1", "id-2"}
    val_ids = {"id-3", "id-4"}
    test_ids = {"id-5", "id-6"}
    findings = validate_split_disjoint(fit_ids, val_ids, test_ids)
    assert findings == [], f"Expected no leakage, got: {findings}"


@sklearn_available
def test_run_fit_leakage_produces_no_artifact(tmp_path: Path) -> None:
    """If leakage is found, run_fit must return no artifact and write no file."""
    from app.calibration.fit import run_fit

    # Pairs with all 3 same identity — any split will have overlap
    pairs = [(0.8, 1, "id-1"), (0.3, 0, "id-1"), (0.75, 1, "id-1")]
    output = tmp_path / "artifact.json"
    artifact, _gate_failures, _leakage = run_fit(
        pairs=pairs,
        method="logistic",
        arcface_model_version="buffalo_l",
        model_profile="full",
        preprocessing_version="v1",
        output_path=output,
    )
    assert artifact is None
    assert not output.exists(), "Artifact must not be written when leakage is found"


# ---------------------------------------------------------------------------
# 9. Failed gates do not emit artifact
# ---------------------------------------------------------------------------


@sklearn_available
def test_failed_gate_produces_no_artifact(tmp_path: Path) -> None:
    """Gates that fail must prevent artifact from being written."""
    # Build pairs that won't satisfy strict gates: random noise
    import random

    from app.calibration.fit import run_fit

    rng = random.Random(42)
    pairs = [
        (rng.uniform(0, 1), rng.randint(0, 1), f"id-{i % 20}")
        for i in range(600)
    ]

    output = tmp_path / "artifact.json"
    # Use impossibly strict gates to force failure
    strict_gates = {
        "max_brier_score": 0.001,
        "max_log_loss": 0.001,
        "max_ece": 0.001,
        "max_fmr_at_operating_point": 0.001,
        "min_fit_pairs": 1,
        "min_val_pairs": 1,
        "min_test_pairs": 1,
        "min_fit_identities": 1,
    }
    artifact, gate_failures, _leakage = run_fit(
        pairs=pairs,
        method="logistic",
        arcface_model_version="buffalo_l",
        model_profile="full",
        preprocessing_version="v1",
        output_path=output,
        gates=strict_gates,
    )
    assert artifact is None
    assert gate_failures, "Expected gate failures"
    assert not output.exists(), "Artifact must not be written when gates fail"


# ---------------------------------------------------------------------------
# 10. check_gates
# ---------------------------------------------------------------------------


def test_check_gates_all_pass() -> None:
    from app.calibration.fit import check_gates

    metrics = CalibrationMetrics(
        brier_score=0.05, log_loss=0.20, ece=0.03,
        fmr_at_operating_point=0.02, fnmr_at_operating_point=0.10,
        operating_point_threshold=0.80,
    )
    counts = SplitCounts(
        fit_pairs=400, val_pairs=100, test_pairs=100, fit_identities=10,
    )
    failures = check_gates(metrics, counts)
    assert failures == []


def test_check_gates_brier_fails() -> None:
    from app.calibration.fit import check_gates

    metrics = CalibrationMetrics(brier_score=0.20)
    counts = SplitCounts(
        fit_pairs=400, val_pairs=100, test_pairs=100, fit_identities=10,
    )
    failures = check_gates(metrics, counts)
    assert any("brier_score" in f for f in failures)


def test_check_gates_insufficient_fit_pairs_fails() -> None:
    from app.calibration.fit import check_gates

    metrics = CalibrationMetrics(
        brier_score=0.05, log_loss=0.15, ece=0.02, fmr_at_operating_point=0.01,
    )
    counts = SplitCounts(
        fit_pairs=10,  # below min_fit_pairs=200
        val_pairs=100, test_pairs=100, fit_identities=10,
    )
    failures = check_gates(metrics, counts)
    assert any("fit_pairs" in f for f in failures)


# ---------------------------------------------------------------------------
# 11. Manifest hash
# ---------------------------------------------------------------------------


def test_hash_dataset_deterministic() -> None:
    from app.calibration.fit import hash_dataset

    pairs = [(0.8, 1), (0.3, 0), (0.5, 1)]
    h1 = hash_dataset(pairs)
    h2 = hash_dataset(pairs)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_dataset_different_for_different_data() -> None:
    from app.calibration.fit import hash_dataset

    p1 = [(0.8, 1), (0.3, 0)]
    p2 = [(0.7, 1), (0.3, 0)]
    assert hash_dataset(p1) != hash_dataset(p2)


# ---------------------------------------------------------------------------
# 12. Health endpoint calibration block (no HTTP, just evaluator accessors)
# ---------------------------------------------------------------------------


def test_degraded_missing_evaluator_accessors() -> None:
    ev = CalibrationEvaluator.from_artifact_path(None)
    assert ev.health() == "degraded_missing"
    assert ev.artifact_version() is None
    assert ev.arcface_model_version() == ""
    assert ev.model_profile() == ""
    assert ev.preprocessing_version() == ""


def test_ready_evaluator_accessors() -> None:
    ev = _make_evaluator()
    assert ev.health() == "ready"
    assert ev.artifact_version() == "test-v1"
    assert ev.arcface_model_version() == "buffalo_l"
    assert ev.model_profile() == "full"
    assert ev.preprocessing_version() == "v1"
