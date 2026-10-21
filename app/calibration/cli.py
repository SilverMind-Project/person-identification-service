"""ArcFace calibration CLI.

Usage:
    uv run --extra calibration-tools python -m app.calibration.cli <command> [options]

Subcommands:
    fit         Fit a calibration artifact from labeled pairs; writes JSON only when all
                gates pass, exits non-zero otherwise.
    evaluate    Evaluate an existing artifact against a labeled pairs dataset.
    inspect     Pretty-print the manifest fields of an artifact without running inference.
    validate    Validate artifact schema, monotonicity, and finite-range checks.
    build-manifest  Turn labeled household captures (and an optional external research set)
                into probe-versus-gallery score records with episode ids. Reads the database
                read-only (DATABASE_URL); an external set needs Triton.
    fit-v2      Fit a schema 2 artifact from a manifest with household-holdout gates; writes JSON
                only when every gate passes.

Requires the calibration-tools optional dependency group (scikit-learn, matplotlib).
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.calibration.manifest import ExternalFace


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
    from app.calibration.evaluator import parse_artifact

    path = Path(args.artifact)
    if not path.exists():
        print(f"FAIL: artifact not found: {path}")
        return 1

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        artifact = parse_artifact(raw)
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
        expected_quality_version=args.quality_version or "",
    )
    state = evaluator.health()
    print(f"calibration_health: {state}")
    print(f"artifact_version:   {evaluator.artifact_version()}")
    print(f"arcface_model:      {evaluator.arcface_model_version()}")
    print(f"model_profile:      {evaluator.model_profile()}")
    print(f"preprocessing:      {evaluator.preprocessing_version()}")
    if evaluator.quality_version():
        print(f"calibration_profile: {evaluator.calibration_profile()}")
        print(f"quality_version:    {evaluator.quality_version()}")
        print(f"covered_identities: {', '.join(evaluator.covered_identities() or []) or '(none)'}")

    if state == "ready":
        print("PASS")
        return 0
    else:
        print("FAIL")
        return 1


async def _build_manifest(args: argparse.Namespace) -> int:
    import asyncpg
    from pgvector.asyncpg import register_vector

    from app import config
    from app.calibration.manifest import (
        GalleryEmbedding,
        ManifestError,
        ManifestMeta,
        external_records,
        household_records,
        load_external_listing,
        write_manifest,
    )
    from app.services.calibration_candidates import PostgresCalibrationCandidateRepository
    from app.services.enrollment_repository import PostgresEnrollmentRepository

    dsn = config.get("database.dsn")
    if not dsn:
        print("FAIL: database.dsn is not configured (set DATABASE_URL)")
        return 1

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    # Read-only: this tool never writes enrollment, captures, or any other table.
    pool = await asyncpg.create_pool(
        dsn,
        init=_init,
        min_size=1,
        max_size=2,
        server_settings={"default_transaction_read_only": "on"},
    )
    try:
        candidates = await PostgresCalibrationCandidateRepository(pool).list_labeled()
        embeddings = await PostgresEnrollmentRepository(pool).member_embeddings()
    finally:
        await pool.close()

    meta = ManifestMeta()
    gallery = {pid: [GalleryEmbedding(e) for e in rows] for pid, rows in embeddings.items()}
    try:
        records = household_records(
            candidates, gallery, episode_gap_s=args.episode_gap_s, meta=meta
        )
        if args.external:
            entries, licence = load_external_listing(Path(args.external))
            faces, dropped = await _detect_external(entries)
            external = external_records(faces)
            records.extend(external)
            meta.external_datasets.append(
                licence
                | {
                    "identities": len({f.identity_id for f in faces}),
                    "images": len(entries),
                    "images_without_face": dropped,
                }
            )
            print(f"external: {len(entries)} images, {dropped} without a usable face")
    except ManifestError as exc:
        print(f"FAIL: {exc}")
        return 1

    output = Path(args.output)
    write_manifest(records, meta, output)
    print(
        f"wrote {len(records)} records ({meta.genuine} genuine, {meta.impostor} impostor) to {output}"
    )
    print(f"probes by identity: {meta.probes_by_identity}; open-set probes: {meta.open_set_probes}")
    print(
        f"skipped: not a face {meta.skipped_not_a_face}, identity not enrolled "
        f"{meta.skipped_identity_not_enrolled}"
    )
    return 0


async def _detect_external(
    entries: list[tuple[Path, str]],
) -> tuple[list[ExternalFace], int]:
    from triton_shared.client import TritonGrpcClient

    from app import config
    from app.calibration.manifest import detect_external_faces
    from app.services.face_engine import FaceEngine
    from app.services.face_quality import build_face_quality_scorer

    client = TritonGrpcClient(
        str(config.get("face_engine.triton_url", "triton:8701")),
        timeout_ms=int(config.get("face_engine.triton_timeout_ms", 30_000)),
    )
    async with client:
        engine = FaceEngine(client, build_face_quality_scorer(config.get("face_quality", {})))
        await engine.validate_models()
        return await detect_external_faces(entries, engine)


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    import asyncio

    return asyncio.run(_build_manifest(args))


def _cmd_fit_v2(args: argparse.Namespace) -> int:
    from app.calibration.fit import DEFAULT_GATES_V2, run_fit_v2
    from app.calibration.manifest import read_manifest

    records, meta = read_manifest(Path(args.manifest))
    gates = dict(DEFAULT_GATES_V2)
    for key in (
        "max_ece",
        "max_brier_score",
        "max_fmr_at_operating_point",
        "min_holdout_genuine_per_identity",
        "min_fit_identities",
    ):
        if getattr(args, key) is not None:
            gates[key] = getattr(args, key)
    result = run_fit_v2(
        records,
        meta,
        model_profile=args.model_profile,
        allow_household_only=args.allow_household_only,
        gates=gates,
        random_seed=args.random_seed,
        output_path=Path(args.output) if args.output else None,
    )
    if result.leakage:
        print("FAIL: leakage findings:")
        for finding in result.leakage:
            print(f"  {finding}")
        return 1
    print(json.dumps(result.report, indent=2, default=str))
    if result.failures:
        print("FAIL: gate failures:")
        for failure in result.failures:
            print(f"  FAIL {failure}")
        return 1
    assert result.artifact is not None
    print(
        f"PASS: artifact {result.artifact.artifact_version} ({result.artifact.fit_basis}), "
        f"covers {result.artifact.covered_identities}"
    )
    if args.output:
        print(f"  artifact written to: {args.output}")
    return 0


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
    p_val.add_argument("--quality-version", dest="quality_version", default=None)

    # -- build-manifest --
    p_manifest = sub.add_parser("build-manifest", help="Build a v2 score manifest from labels")
    p_manifest.add_argument("--output", required=True, help="manifest .jsonl path (private data)")
    p_manifest.add_argument("--external", default=None, help="external set JSONL (needs Triton)")
    p_manifest.add_argument("--episode-gap-s", dest="episode_gap_s", type=float, default=300.0)

    # -- fit-v2 --
    p_v2 = sub.add_parser("fit-v2", help="Fit a schema 2 artifact with household-holdout gates")
    p_v2.add_argument("--manifest", required=True, help="manifest .jsonl from build-manifest")
    p_v2.add_argument("--output", default=None, help="artifact path; omitted means dry run")
    p_v2.add_argument("--model-profile", dest="model_profile", default="full")
    p_v2.add_argument("--random-seed", dest="random_seed", type=int, default=42)
    p_v2.add_argument(
        "--allow-household-only",
        dest="allow_household_only",
        action="store_true",
        help="fit without an external set (allowed for a household deployment; recorded)",
    )
    p_v2.add_argument("--max-ece", dest="max_ece", type=float, default=None)
    p_v2.add_argument("--max-brier-score", dest="max_brier_score", type=float, default=None)
    p_v2.add_argument("--max-fmr", dest="max_fmr_at_operating_point", type=float, default=None)
    p_v2.add_argument(
        "--min-holdout-genuine", dest="min_holdout_genuine_per_identity", type=int, default=None
    )
    p_v2.add_argument("--min-fit-identities", dest="min_fit_identities", type=int, default=None)

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
        case "build-manifest":
            return _cmd_build_manifest(ns)
        case "fit-v2":
            return _cmd_fit_v2(ns)
        case _:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
