"""ArcFace calibration CLI.

Usage:
    uv run --extra calibration-tools python -m app.calibration.cli <command> [options]

Subcommands:
    fit         Fit a calibration artifact from labeled pairs; writes JSON only when all
                gates pass, exits non-zero otherwise.
    evaluate    Evaluate an existing artifact against a labeled pairs dataset.
    inspect     Pretty-print the manifest fields of an artifact without running inference.
    validate    Validate artifact schema, monotonicity, and finite-range checks.

Requires the calibration-tools optional dependency group (scikit-learn, matplotlib).
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_fit(args: argparse.Namespace) -> int:
    """Fit a calibration artifact. Exits non-zero if gates fail or leakage found."""
    from app.calibration.fit import DEFAULT_GATES, load_pairs_json, run_fit

    pairs = load_pairs_json(Path(args.pairs))
    print(f"Loaded {len(pairs)} pairs from {args.pairs}")

    gates: dict[str, float] = dict(DEFAULT_GATES)
    if args.max_brier_score is not None:
        gates["max_brier_score"] = args.max_brier_score
    if args.max_log_loss is not None:
        gates["max_log_loss"] = args.max_log_loss
    if args.max_ece is not None:
        gates["max_ece"] = args.max_ece
    if args.max_fmr is not None:
        gates["max_fmr_at_operating_point"] = args.max_fmr
    if args.min_fit_pairs is not None:
        gates["min_fit_pairs"] = args.min_fit_pairs
    if args.operating_point is not None:
        gates["operating_point_threshold"] = args.operating_point

    output = Path(args.output) if args.output else None

    artifact, gate_failures, leakage_findings = run_fit(
        pairs=pairs,
        method=args.method,
        arcface_model_version=args.arcface_model_version,
        model_profile=args.model_profile,
        preprocessing_version=args.preprocessing_version,
        output_path=output,
        gates=gates,
        random_seed=args.random_seed,
        fit_frac=args.fit_frac,
        val_frac=args.val_frac,
    )

    if leakage_findings:
        print("FAIL: leakage findings:")
        for f in leakage_findings:
            print(f"  {f}")
        return 1

    if gate_failures:
        print("FAIL: gate failures:")
        for f in gate_failures:
            print(f"  FAIL {f}")
        return 1

    if artifact is None:
        print("FAIL: artifact not produced (unknown error)")
        return 1

    m = artifact.metrics
    sc = artifact.split_counts
    print("PASS: all gates satisfied")
    print(f"  method:               {artifact.method}")
    print(f"  artifact_version:     {artifact.artifact_version}")
    print(f"  arcface_model:        {artifact.arcface_model_version}")
    print(f"  model_profile:        {artifact.model_profile}")
    print(f"  preprocessing:        {artifact.preprocessing_version}")
    print(f"  fit_pairs:            {sc.fit_pairs} (+{sc.fit_positives} / -{sc.fit_negatives})")
    print(f"  val_pairs:            {sc.val_pairs} (+{sc.val_positives} / -{sc.val_negatives})")
    print(f"  test_pairs:           {sc.test_pairs} (+{sc.test_positives} / -{sc.test_negatives})")
    print(f"  fit_identities:       {sc.fit_identities}")
    print(f"  brier_score:          {m.brier_score}")
    print(f"  log_loss:             {m.log_loss}")
    print(f"  ece:                  {m.ece}")
    print(f"  fmr_at_op_point:      {m.fmr_at_operating_point}")
    print(f"  fnmr_at_op_point:     {m.fnmr_at_operating_point}")
    print(f"  operating_point:      {m.operating_point_threshold}")
    print(f"  fit_dataset_hash:     {artifact.fit_dataset_hash}")
    print(f"  val_dataset_hash:     {artifact.val_dataset_hash}")
    print(f"  test_dataset_hash:    {artifact.test_dataset_hash}")
    if output:
        print(f"  artifact written to:  {output}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate an existing artifact against a labeled pairs dataset."""
    from app.calibration.evaluator import CalibrationEvaluator
    from app.calibration.fit import compute_metrics, load_pairs_json
    from app.calibration.models import CalibrationArtifact

    path = Path(args.artifact)
    if not path.exists():
        print(f"FAIL: artifact not found: {path}")
        return 1

    evaluator = CalibrationEvaluator.from_artifact_path(
        args.artifact,
        expected_arcface_model_version=args.arcface_model_version or "",
        expected_model_profile=args.model_profile or "",
        expected_preprocessing_version=args.preprocessing_version or "",
    )
    if evaluator.health() != "ready":
        print(f"FAIL: calibration_health={evaluator.health()}")
        return 1

    artifact = CalibrationArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
    pairs = load_pairs_json(Path(args.pairs))
    eval_pairs = [(s, lbl) for s, lbl, _ in pairs]

    operating_point = float(args.operating_point or 0.80)
    metrics = compute_metrics(
        eval_pairs,
        artifact.coef,
        artifact.intercept,
        artifact.x_knots,
        artifact.y_knots,
        artifact.method,
        operating_point,
    )

    print(f"artifact:           {args.artifact}")
    print(f"artifact_version:   {artifact.artifact_version}")
    print(f"pairs evaluated:    {len(eval_pairs)}")
    print(f"brier_score:        {metrics.brier_score}")
    print(f"log_loss:           {metrics.log_loss}")
    print(f"ece:                {metrics.ece}")
    print(f"fmr_at_op_point:    {metrics.fmr_at_operating_point}")
    print(f"fnmr_at_op_point:   {metrics.fnmr_at_operating_point}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Pretty-print artifact manifest fields."""
    from app.calibration.models import CalibrationArtifact

    path = Path(args.artifact)
    if not path.exists():
        print(f"FAIL: artifact not found: {path}")
        return 1

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        artifact = CalibrationArtifact.model_validate(raw)
    except Exception as exc:
        print(f"FAIL: could not parse artifact: {exc}")
        return 1

    print(json.dumps(artifact.model_dump(mode="json"), indent=2, default=str))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate artifact schema, monotonicity, and finite-range checks."""
    from app.calibration.evaluator import CalibrationEvaluator

    evaluator = CalibrationEvaluator.from_artifact_path(
        args.artifact,
        expected_arcface_model_version=args.arcface_model_version or "",
        expected_model_profile=args.model_profile or "",
        expected_preprocessing_version=args.preprocessing_version or "",
    )
    state = evaluator.health()
    print(f"calibration_health: {state}")
    print(f"artifact_version:   {evaluator.artifact_version()}")
    print(f"arcface_model:      {evaluator.arcface_model_version()}")
    print(f"model_profile:      {evaluator.model_profile()}")
    print(f"preprocessing:      {evaluator.preprocessing_version()}")

    if state == "ready":
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.calibration.cli",
        description="ArcFace confidence calibration toolchain",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- fit --
    p_fit = sub.add_parser("fit", help="Fit a calibration artifact from labeled pairs")
    p_fit.add_argument("--pairs", required=True, help="Path to pairs JSON file")
    p_fit.add_argument("--output", default=None, help="Output artifact path (.json)")
    p_fit.add_argument(
        "--method", choices=["logistic", "isotonic"], default="logistic", help="Calibration method"
    )
    p_fit.add_argument("--arcface-model-version", dest="arcface_model_version", default="buffalo_l")
    p_fit.add_argument("--model-profile", dest="model_profile", default="full")
    p_fit.add_argument("--preprocessing-version", dest="preprocessing_version", default="v1")
    p_fit.add_argument("--random-seed", dest="random_seed", type=int, default=42)
    p_fit.add_argument("--fit-frac", dest="fit_frac", type=float, default=0.6)
    p_fit.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    p_fit.add_argument("--max-brier-score", dest="max_brier_score", type=float, default=None)
    p_fit.add_argument("--max-log-loss", dest="max_log_loss", type=float, default=None)
    p_fit.add_argument("--max-ece", dest="max_ece", type=float, default=None)
    p_fit.add_argument("--max-fmr", dest="max_fmr", type=float, default=None)
    p_fit.add_argument("--min-fit-pairs", dest="min_fit_pairs", type=int, default=None)
    p_fit.add_argument("--operating-point", dest="operating_point", type=float, default=None)

    # -- evaluate --
    p_eval = sub.add_parser("evaluate", help="Evaluate an artifact against a dataset")
    p_eval.add_argument("--artifact", required=True, help="Path to artifact JSON")
    p_eval.add_argument("--pairs", required=True, help="Path to pairs JSON file")
    p_eval.add_argument("--arcface-model-version", dest="arcface_model_version", default=None)
    p_eval.add_argument("--model-profile", dest="model_profile", default=None)
    p_eval.add_argument("--preprocessing-version", dest="preprocessing_version", default=None)
    p_eval.add_argument("--operating-point", dest="operating_point", type=float, default=0.80)

    # -- inspect --
    p_inspect = sub.add_parser("inspect", help="Pretty-print artifact manifest")
    p_inspect.add_argument("--artifact", required=True, help="Path to artifact JSON")

    # -- validate --
    p_val = sub.add_parser("validate", help="Validate artifact schema and checks")
    p_val.add_argument("--artifact", required=True, help="Path to artifact JSON")
    p_val.add_argument("--arcface-model-version", dest="arcface_model_version", default=None)
    p_val.add_argument("--model-profile", dest="model_profile", default=None)
    p_val.add_argument("--preprocessing-version", dest="preprocessing_version", default=None)

    ns = parser.parse_args(argv)
    match ns.command:
        case "fit":
            return _cmd_fit(ns)
        case "evaluate":
            return _cmd_evaluate(ns)
        case "inspect":
            return _cmd_inspect(ns)
        case "validate":
            return _cmd_validate(ns)
        case _:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
