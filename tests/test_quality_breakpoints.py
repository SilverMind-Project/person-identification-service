"""Offline face quality breakpoint analysis and the calibration script's manifest loader."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")

from app.calibration.quality_breakpoints import FaceQualitySample, analyze

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_face_quality.py"


def _samples() -> list[FaceQualitySample]:
    # Larger faces and flatter pitch match better; detection score carries no signal (constant).
    return [
        FaceQualitySample(
            interocular_px=8.0 + 2.0 * i,
            det_score=0.8,
            abs_pitch_deg=40.0 - 2.0 * i,
            log10_laplacian_variance=1.5 + 0.05 * i,
            raw_similarity=0.2 + 0.03 * i,
        )
        for i in range(20)
    ]


def _by_name(samples: list[FaceQualitySample]) -> dict[str, object]:
    return {report.name: report for report in analyze(samples)}


def test_correlations_have_the_expected_sign() -> None:
    reports = _by_name(_samples())
    assert reports["interocular_px"].spearman_rho == pytest.approx(1.0)
    assert reports["abs_pitch_deg"].spearman_rho == pytest.approx(-1.0)
    assert reports["log10_laplacian_variance"].spearman_rho == pytest.approx(1.0)


def test_a_constant_input_has_no_correlation() -> None:
    det = _by_name(_samples())["det_score"]
    assert det.spearman_rho is None
    assert det.spearman_p_value is None


def test_anchors_are_the_5th_and_95th_percentiles_with_pitch_descending() -> None:
    reports = _by_name(_samples())
    iod = reports["interocular_px"]
    assert (iod.suggested_low, iod.suggested_high) == pytest.approx((9.9, 44.1))
    pitch = reports["abs_pitch_deg"]
    assert (pitch.suggested_low, pitch.suggested_high) == pytest.approx((38.1, 3.9))


def test_unmeasured_interocular_is_skipped_for_that_input_only() -> None:
    samples = _samples()
    samples[0] = FaceQualitySample(None, 0.8, 40.0, 1.5, 0.2)
    reports = _by_name(samples)
    assert reports["interocular_px"].count == 19
    assert reports["det_score"].count == 20


def test_no_samples_is_an_error() -> None:
    with pytest.raises(ValueError, match="no labeled faces"):
        analyze([])


def _load_manifest():
    spec = importlib.util.spec_from_file_location("calibrate_face_quality", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._load_manifest


def test_manifest_resolves_images_relative_to_the_manifest(tmp_path: Path) -> None:
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "001.jpg").write_bytes(b"x")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "faces": [{"image": "alpha/001.jpg", "identity_id": "a"}]})
    )
    assert _load_manifest()(manifest) == [(tmp_path / "alpha" / "001.jpg", "a")]


def test_manifest_rejects_wrong_schema_and_missing_images(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "faces": []}))
    with pytest.raises(ValueError, match="schema_version"):
        _load_manifest()(manifest)
    manifest.write_text(
        json.dumps({"schema_version": 1, "faces": [{"image": "gone.jpg", "identity_id": "a"}]})
    )
    with pytest.raises(FileNotFoundError, match=r"gone\.jpg"):
        _load_manifest()(manifest)
