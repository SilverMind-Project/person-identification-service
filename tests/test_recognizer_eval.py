"""Tests for Part B recognizer evaluation determinism, report schema, and adoption gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.calibration.recognizer_eval import (
    CANDIDATE_LICENCE_AUDIT,
    INTEROCULAR_BANDS,
    HoldoutProbeScore,
    RecognizerComparisonReport,
    evaluate_recognizer_comparison,
)
from scripts.evaluate_recognizer_swap import run_comparison


def _synthetic_holdout(
    *,
    profile_name: str,
    recognizer_version: str,
    gar_16_32_boost: float = 0.0,
    open_set_shift: float = 0.0,
    cal_bias: float = 0.0,
) -> dict[str, Any]:
    """Build a deterministic household holdout payload covering all interocular bands."""
    probes: list[dict[str, Any]] = []
    # 200 open-set impostor probes with similarities linearly spanning [0.05, 0.45] + shift.
    for i in range(200):
        sim = 0.05 + 0.40 * (i / 199.0) + open_set_shift
        prob = min(1.0, max(0.0, 0.02 + cal_bias))
        probes.append(
            {
                "probe_id": f"imp-{i}",
                "identity_id": None,
                "similarity": round(sim, 6),
                "calibrated_probability": round(prob, 6),
                "is_genuine": False,
                "is_open_set": True,
                "interocular_px": 22.0 if i % 2 == 0 else 36.0,
            }
        )
    # 40 genuine probes in 16-32 px band (half in 16-24, half in 24-32), plus <16 and >=40 bands.
    band_specs = [
        ("lt16", 12.0, 10, 0.30),
        ("mid1", 20.0, 20, 0.42 + gar_16_32_boost),
        ("mid2", 28.0, 20, 0.46 + gar_16_32_boost),
        ("hi", 45.0, 20, 0.78),
    ]
    for tag, iod, count, base_sim in band_specs:
        for i in range(count):
            # Spread genuine scores cleanly across the operating threshold.
            sim = base_sim + (i - count / 2) * 0.015
            prob = min(1.0, max(0.0, 0.96 - cal_bias))
            probes.append(
                {
                    "probe_id": f"gen-{tag}-{i}",
                    "identity_id": "resident-alpha" if i % 2 == 0 else "resident-beta",
                    "similarity": round(sim, 6),
                    "calibrated_probability": round(prob, 6),
                    "is_genuine": True,
                    "is_open_set": False,
                    "interocular_px": iod,
                }
            )
    return {
        "profile_name": profile_name,
        "recognizer_version": recognizer_version,
        "covered_identities": ["resident-alpha", "resident-beta"],
        "probes": probes,
    }


def test_evaluation_script_is_deterministic_and_matches_report_schema(tmp_path: Path) -> None:
    base_file = tmp_path / "base.json"
    cand_file = tmp_path / "cand.json"
    base_file.write_text(
        json.dumps(
            _synthetic_holdout(
                profile_name="buffalo_l",
                recognizer_version="buffalo_l",
                gar_16_32_boost=0.0,
            )
        ),
        encoding="utf-8",
    )
    cand_file.write_text(
        json.dumps(
            _synthetic_holdout(
                profile_name="adaface_ir50_webface4m",
                recognizer_version="adaface_ir50_webface4m",
                gar_16_32_boost=0.12,
            )
        ),
        encoding="utf-8",
    )

    out_1 = tmp_path / "report_1.json"
    out_2 = tmp_path / "report_2.json"
    r1 = run_comparison(base_file, cand_file, output_path=out_1, random_seed=42)
    r2 = run_comparison(base_file, cand_file, output_path=out_2, random_seed=42)

    raw_1 = out_1.read_text(encoding="utf-8")
    raw_2 = out_2.read_text(encoding="utf-8")
    assert raw_1 == raw_2
    assert r1 == r2

    parsed = RecognizerComparisonReport.model_validate_json(raw_1)
    assert parsed.schema_version == 1
    assert set(parsed.baseline.by_interocular_band) == {name for name, _, _ in INTEROCULAR_BANDS}
    assert set(parsed.candidate.by_interocular_band) == {name for name, _, _ in INTEROCULAR_BANDS}
    assert parsed.baseline_licence.commercial_clean is False
    assert parsed.candidate_licence.commercial_clean is False
    assert "WebFace4M" in parsed.candidate_licence.training_data_licence
    assert parsed.decision.decision == "adopt"
    assert parsed.decision.gain_16_to_32px_pp is not None
    assert parsed.decision.gain_16_to_32px_pp >= 10.0


def test_adoption_gate_rejects_small_gain_and_poor_ece(tmp_path: Path) -> None:
    base_file = tmp_path / "base.json"
    cand_small = tmp_path / "cand_small.json"
    cand_bad_ece = tmp_path / "cand_bad_ece.json"

    base_file.write_text(
        json.dumps(
            _synthetic_holdout(
                profile_name="buffalo_l",
                recognizer_version="buffalo_l",
                gar_16_32_boost=0.0,
            )
        ),
        encoding="utf-8",
    )
    cand_small.write_text(
        json.dumps(
            _synthetic_holdout(
                profile_name="adaface_ir50_webface4m",
                recognizer_version="adaface_ir50_webface4m",
                gar_16_32_boost=0.01,
            )
        ),
        encoding="utf-8",
    )
    cand_bad_ece.write_text(
        json.dumps(
            _synthetic_holdout(
                profile_name="adaface_ir50_webface4m",
                recognizer_version="adaface_ir50_webface4m",
                gar_16_32_boost=0.15,
                cal_bias=0.35,
            )
        ),
        encoding="utf-8",
    )

    small_report = run_comparison(base_file, cand_small)
    assert small_report.decision.decision == "keep_baseline"
    assert any("below required" in r for r in small_report.decision.reasons)

    bad_ece_report = run_comparison(base_file, cand_bad_ece)
    assert bad_ece_report.decision.decision == "keep_baseline"
    assert bad_ece_report.decision.ece_ok is False
    assert any("ECE" in r for r in bad_ece_report.decision.reasons)


def test_adoption_gate_returns_insufficient_data_without_open_set_probes() -> None:
    only_genuine = [
        HoldoutProbeScore(
            probe_id="g1",
            identity_id="resident-alpha",
            similarity=0.8,
            calibrated_probability=0.95,
            is_genuine=True,
            is_open_set=False,
            interocular_px=22.0,
        )
    ]
    report = evaluate_recognizer_comparison(
        baseline_profile="buffalo_l",
        baseline_recognizer_version="buffalo_l",
        baseline_probes=only_genuine,
        baseline_covered_identities=["resident-alpha"],
        candidate_profile="adaface_ir50_webface4m",
        candidate_recognizer_version="adaface_ir50_webface4m",
        candidate_probes=only_genuine,
        candidate_covered_identities=["resident-alpha"],
    )
    assert report.decision.decision == "insufficient_data"
    assert any("open-set" in r for r in report.decision.reasons)


def test_candidate_licence_audit_records_code_weights_and_training_data() -> None:
    assert {"buffalo_l", "adaface_ir50_webface4m", "opencv_sface_2021dec", "dlib_resnet_v1"} <= set(
        CANDIDATE_LICENCE_AUDIT
    )
    for rec in CANDIDATE_LICENCE_AUDIT.values():
        assert rec.code_licence
        assert rec.weights_licence
        assert rec.training_data_licence
        assert rec.embedding_dim > 0
        # None of the public research-dataset checkpoints are commercially clean on training data.
        assert rec.commercial_clean is False


def test_published_comparison_report_schema_matches_model() -> None:
    published = json.loads(
        (
            Path(__file__).parent.parent / "schemas/recognizer_comparison_report.schema.json"
        ).read_text(encoding="utf-8")
    )
    expected = RecognizerComparisonReport.model_json_schema()
    expected["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    expected["$id"] = "recognizer_comparison_report.schema.json"
    assert published == expected
