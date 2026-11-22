"""Focused tests for narrowed exception handling in calibration CLI and evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from app.calibration.cli import _cmd_inspect
from app.calibration.evaluator import CalibrationEvaluator


def test_cli_inspect_handles_malformed_json_and_schema_errors(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert _cmd_inspect(argparse.Namespace(artifact=str(bad_json))) == 1

    bad_schema = tmp_path / "bad_schema.json"
    bad_schema.write_text('{"schema_version": 99}', encoding="utf-8")
    assert _cmd_inspect(argparse.Namespace(artifact=str(bad_schema))) == 1


def test_evaluator_predict_catches_arithmetic_error_and_propagates_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "identity_integrity"
        / "calibration_artifact_logistic.json"
    )
    evaluator = CalibrationEvaluator.from_artifact_path(artifact_path)
    assert evaluator.health() == "ready"

    def _raise_value_error(_artifact: object, _features: object) -> float:
        raise ValueError("bad numeric feature")

    monkeypatch.setattr("app.calibration.evaluator._apply", _raise_value_error)
    assert evaluator.predict({"similarity": 0.5}, "alice") is None

    def _raise_programmer_bug(_artifact: object, _features: object) -> float:
        raise RuntimeError("unexpected programming error")

    monkeypatch.setattr("app.calibration.evaluator._apply", _raise_programmer_bug)
    with pytest.raises(RuntimeError, match="unexpected programming error"):
        evaluator.predict({"similarity": 0.5}, "alice")
