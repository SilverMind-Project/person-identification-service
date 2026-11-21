"""Offline CLI comparing a candidate face recognizer against the baseline on household holdout.

Usage:
    uv run --extra calibration-tools python scripts/evaluate_recognizer_swap.py \\
        --baseline-scores baseline_holdout.json \\
        --candidate-scores candidate_holdout.json \\
        --output comparison_report.json

Input files are JSON objects with:
    {
      "profile_name": "buffalo_l",
      "recognizer_version": "buffalo_l",
      "covered_identities": ["resident-alpha", "resident-beta"],
      "probes": [
        {
          "probe_id": "p1",
          "identity_id": "resident-alpha",
          "similarity": 0.72,
          "calibrated_probability": 0.91,
          "is_genuine": true,
          "is_open_set": false,
          "interocular_px": 22.5
        }
      ]
    }

Deterministic for identical inputs and seed. Writes schema-validated JSON only.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.calibration.recognizer_eval import (
    DEFAULT_MAX_ECE,
    DEFAULT_MAX_OPEN_SET_FMR,
    DEFAULT_MIN_GAIN_16_TO_32PX_PP,
    HoldoutProbeScore,
    RecognizerComparisonReport,
    evaluate_recognizer_comparison,
)


def _load_holdout(path: Path) -> tuple[str, str, list[str], list[HoldoutProbeScore]]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    probes = [
        HoldoutProbeScore(
            probe_id=str(item["probe_id"]),
            identity_id=None if item.get("identity_id") is None else str(item["identity_id"]),
            similarity=float(item["similarity"]),
            calibrated_probability=(
                None
                if item.get("calibrated_probability") is None
                else float(item["calibrated_probability"])
            ),
            is_genuine=bool(item["is_genuine"]),
            is_open_set=bool(item["is_open_set"]),
            interocular_px=(
                None if item.get("interocular_px") is None else float(item["interocular_px"])
            ),
        )
        for item in raw["probes"]
    ]
    return (
        str(raw["profile_name"]),
        str(raw["recognizer_version"]),
        [str(i) for i in raw["covered_identities"]],
        probes,
    )


def run_comparison(
    baseline_path: Path,
    candidate_path: Path,
    *,
    output_path: Path | None = None,
    random_seed: int = 42,
    max_open_set_fmr: float = DEFAULT_MAX_OPEN_SET_FMR,
    min_gain_16_to_32px_pp: float = DEFAULT_MIN_GAIN_16_TO_32PX_PP,
    max_ece: float = DEFAULT_MAX_ECE,
) -> RecognizerComparisonReport:
    b_prof, b_ver, b_cov, b_probes = _load_holdout(baseline_path)
    c_prof, c_ver, c_cov, c_probes = _load_holdout(candidate_path)
    report = evaluate_recognizer_comparison(
        baseline_profile=b_prof,
        baseline_recognizer_version=b_ver,
        baseline_probes=b_probes,
        baseline_covered_identities=b_cov,
        candidate_profile=c_prof,
        candidate_recognizer_version=c_ver,
        candidate_probes=c_probes,
        candidate_covered_identities=c_cov,
        random_seed=random_seed,
        max_open_set_fmr=max_open_set_fmr,
        min_gain_16_to_32px_pp=min_gain_16_to_32px_pp,
        max_ece=max_ece,
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Canonical sorted keys for byte-for-byte deterministic output.
        rendered = json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        output_path.write_text(rendered, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate face recognizers on household holdout scores"
    )
    parser.add_argument("--baseline-scores", required=True, type=Path)
    parser.add_argument("--candidate-scores", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-open-set-fmr", type=float, default=DEFAULT_MAX_OPEN_SET_FMR)
    parser.add_argument(
        "--min-gain-16-to-32px-pp", type=float, default=DEFAULT_MIN_GAIN_16_TO_32PX_PP
    )
    parser.add_argument("--max-ece", type=float, default=DEFAULT_MAX_ECE)
    args = parser.parse_args(argv)

    report = run_comparison(
        args.baseline_scores,
        args.candidate_scores,
        output_path=args.output,
        random_seed=args.random_seed,
        max_open_set_fmr=args.max_open_set_fmr,
        min_gain_16_to_32px_pp=args.min_gain_16_to_32px_pp,
        max_ece=args.max_ece,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
