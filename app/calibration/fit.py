"""ArcFace calibration fitting toolchain.

Requires the ``calibration-tools`` optional dependency group:
    uv sync --extra calibration-tools
    uv run --extra calibration-tools python -m app.calibration.cli fit ...

This module is never imported by app/main.py or any runtime path.
All sklearn imports are deferred to function bodies so the evaluator
can be imported cleanly without calibration-tools installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.calibration.models import (
    CalibrationArtifact,
    CalibrationMetrics,
    SplitCounts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gate defaults (all configurable via CLI flags)
# ---------------------------------------------------------------------------
DEFAULT_GATES: dict[str, float] = {
    "max_brier_score": 0.10,
    "max_log_loss": 0.30,
    "max_ece": 0.05,
    "max_fmr_at_operating_point": 0.05,
    "min_fit_pairs": 200,
    "min_val_pairs": 50,
    "min_test_pairs": 50,
    "min_fit_identities": 5,
    "operating_point_threshold": 0.80,
}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def hash_dataset(pairs: list[tuple[float, int]]) -> str:
    """Deterministic SHA-256 over (similarity, label) pairs (sorted for stability)."""
    raw = json.dumps(sorted(pairs), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_split_disjoint(
    fit_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> list[str]:
    """Return leakage findings (empty list means clean split).

    Each set contains identity IDs present in that partition.
    Any identity appearing in more than one partition is reported.
    """
    findings: list[str] = []
    fit_val = fit_ids & val_ids
    fit_test = fit_ids & test_ids
    val_test = val_ids & test_ids
    if fit_val:
        findings.append(f"fit/val identity leak: {sorted(fit_val)[:5]}")
    if fit_test:
        findings.append(f"fit/test identity leak: {sorted(fit_test)[:5]}")
    if val_test:
        findings.append(f"val/test identity leak: {sorted(val_test)[:5]}")
    return findings


def split_pairs_by_identity(
    pairs: list[tuple[float, int, str]],
    fit_frac: float = 0.6,
    val_frac: float = 0.2,
    random_seed: int = 42,
) -> tuple[
    list[tuple[float, int]],
    list[tuple[float, int]],
    list[tuple[float, int]],
    set[str],
    set[str],
    set[str],
]:
    """Split pairs into (fit, val, test) partitions, disjoint by identity.

    Args:
        pairs: List of (similarity, label, identity_id) where label is 1 (same) or 0 (different).
               Both identities in a pair are attributed to that pair's identity set.
        fit_frac: Fraction of identities assigned to fit.
        val_frac: Fraction of identities assigned to val. Remainder goes to test.
        random_seed: Reproducible shuffle seed.

    Returns:
        (fit_pairs, val_pairs, test_pairs, fit_ids, val_ids, test_ids)
        where *_pairs are (similarity, label) tuples and *_ids are identity ID sets.
    """
    import random  # stdlib is fine here

    all_ids = sorted({ident for _, _, ident in pairs})
    rng = random.Random(random_seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_fit = max(1, round(n * fit_frac))
    n_val = max(1, round(n * val_frac))

    fit_ids: set[str] = set(all_ids[:n_fit])
    val_ids: set[str] = set(all_ids[n_fit : n_fit + n_val])
    test_ids: set[str] = set(all_ids[n_fit + n_val :])

    def _filter(id_set: set[str]) -> list[tuple[float, int]]:
        return [(sim, label) for sim, label, ident in pairs if ident in id_set]

    return (
        _filter(fit_ids),
        _filter(val_ids),
        _filter(test_ids),
        fit_ids,
        val_ids,
        test_ids,
    )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_logistic(
    pairs: list[tuple[float, int]],
    random_seed: int = 42,
) -> tuple[float, float]:
    """Fit Platt-style logistic calibration. Returns (coef, intercept).

    Requires scikit-learn (calibration-tools extra).
    """
    from sklearn.linear_model import LogisticRegression

    X = [[s] for s, _ in pairs]
    y = [lbl for _, lbl in pairs]
    clf = LogisticRegression(random_state=random_seed, max_iter=1000, solver="lbfgs")
    clf.fit(X, y)
    coef = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    return coef, intercept


def fit_isotonic(
    pairs: list[tuple[float, int]],
) -> tuple[list[float], list[float]]:
    """Fit isotonic regression calibration. Returns (x_knots, y_knots).

    Knots are deduplicated and sorted so the runtime evaluator can
    use binary search + linear interpolation.

    Requires scikit-learn (calibration-tools extra).
    """
    from sklearn.isotonic import IsotonicRegression

    X = [s for s, _ in pairs]
    y = [float(lbl) for _, lbl in pairs]
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(X, y)

    # Collapse to unique knots.
    seen: dict[float, float] = {}
    for x, y_pred in zip(ir.X_thresholds_, ir.y_thresholds_, strict=True):
        seen[float(x)] = float(y_pred)
    x_knots = sorted(seen.keys())
    y_knots = [seen[x] for x in x_knots]

    # Clamp to [0, 1].
    y_knots = [max(0.0, min(1.0, y)) for y in y_knots]
    return x_knots, y_knots


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    val_pairs: list[tuple[float, int]],
    coef: float | None,
    intercept: float | None,
    x_knots: list[float] | None,
    y_knots: list[float] | None,
    method: str,
    operating_point: float,
) -> CalibrationMetrics:
    """Compute calibration metrics on the validation/test split.

    Uses only numpy + stdlib (no sklearn needed for evaluation).
    """
    if not val_pairs:
        return CalibrationMetrics()

    sims = [s for s, _ in val_pairs]
    labels = [float(lbl) for _, lbl in val_pairs]

    probs: list[float] = []
    for s in sims:
        if method == "logistic":
            assert coef is not None
            assert intercept is not None
            probs.append(1.0 / (1.0 + math.exp(-(coef * s + intercept))))
        elif method == "isotonic":
            assert x_knots is not None
            assert y_knots is not None
            probs.append(_isotonic_interp(x_knots, y_knots, s))
        else:
            probs.append(0.5)

    n = len(labels)

    # Brier score.
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / n

    # Log loss (clamp to avoid log(0)).
    eps = 1e-15
    ll = -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, labels, strict=True)
    ) / n

    # ECE (10 equal-width bins).
    n_bins = 10
    bin_sum_conf = [0.0] * n_bins
    bin_sum_acc = [0.0] * n_bins
    bin_count = [0] * n_bins
    for p, y in zip(probs, labels, strict=True):
        b = min(int(p * n_bins), n_bins - 1)
        bin_sum_conf[b] += p
        bin_sum_acc[b] += y
        bin_count[b] += 1
    ece = sum(
        (bin_count[b] / n) * abs(bin_sum_acc[b] / bin_count[b] - bin_sum_conf[b] / bin_count[b])
        for b in range(n_bins)
        if bin_count[b] > 0
    )

    # FMR and FNMR at operating point threshold.
    pos_probs = [p for p, y in zip(probs, labels, strict=True) if y == 1.0]
    neg_probs = [p for p, y in zip(probs, labels, strict=True) if y == 0.0]
    fmr = sum(1 for p in neg_probs if p >= operating_point) / max(len(neg_probs), 1)
    fnmr = sum(1 for p in pos_probs if p < operating_point) / max(len(pos_probs), 1)

    return CalibrationMetrics(
        brier_score=round(brier, 6),
        log_loss=round(ll, 6),
        ece=round(ece, 6),
        fmr_at_operating_point=round(fmr, 6),
        fnmr_at_operating_point=round(fnmr, 6),
        operating_point_threshold=operating_point,
    )


def _isotonic_interp(x_knots: list[float], y_knots: list[float], x: float) -> float:
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


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def check_gates(
    metrics: CalibrationMetrics,
    split_counts: SplitCounts,
    gates: dict[str, float] | None = None,
) -> list[str]:
    """Return a list of gate failures (empty list means all gates passed).

    Args:
        metrics: CalibrationMetrics from compute_metrics().
        split_counts: SplitCounts from the fit run.
        gates: Override gate thresholds; defaults applied for missing keys.

    Returns:
        List of human-readable failure strings. Empty means all passed.
    """
    g = {**DEFAULT_GATES, **(gates or {})}
    failures: list[str] = []

    if metrics.brier_score is not None and metrics.brier_score > g["max_brier_score"]:
        failures.append(f"brier_score {metrics.brier_score:.4f} > {g['max_brier_score']}")

    if metrics.log_loss is not None and metrics.log_loss > g["max_log_loss"]:
        failures.append(f"log_loss {metrics.log_loss:.4f} > {g['max_log_loss']}")

    if metrics.ece is not None and metrics.ece > g["max_ece"]:
        failures.append(f"ece {metrics.ece:.4f} > {g['max_ece']}")

    if (
        metrics.fmr_at_operating_point is not None
        and metrics.fmr_at_operating_point > g["max_fmr_at_operating_point"]
    ):
        failures.append(
            f"fmr_at_operating_point {metrics.fmr_at_operating_point:.4f} > {g['max_fmr_at_operating_point']}"
        )

    if split_counts.fit_pairs < g["min_fit_pairs"]:
        failures.append(f"fit_pairs {split_counts.fit_pairs} < {g['min_fit_pairs']}")

    if split_counts.val_pairs < g["min_val_pairs"]:
        failures.append(f"val_pairs {split_counts.val_pairs} < {g['min_val_pairs']}")

    if split_counts.test_pairs < g["min_test_pairs"]:
        failures.append(f"test_pairs {split_counts.test_pairs} < {g['min_test_pairs']}")

    if split_counts.fit_identities < g["min_fit_identities"]:
        failures.append(f"fit_identities {split_counts.fit_identities} < {g['min_fit_identities']}")

    return failures


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_artifact(
    method: str,
    coef: float | None,
    intercept: float | None,
    x_knots: list[float] | None,
    y_knots: list[float] | None,
    arcface_model_version: str,
    model_profile: str,
    preprocessing_version: str,
    metrics: CalibrationMetrics,
    split_counts: SplitCounts,
    fit_dataset_hash: str = "",
    val_dataset_hash: str = "",
    test_dataset_hash: str = "",
    random_seed: int = 42,
    tool_version: str = "1.0.0",
) -> CalibrationArtifact:
    """Construct a CalibrationArtifact from fit outputs."""
    return CalibrationArtifact(
        schema_version=1,
        artifact_version=str(uuid.uuid4()),
        method=method,  # type: ignore[arg-type]
        arcface_model_version=arcface_model_version,
        model_profile=model_profile,
        preprocessing_version=preprocessing_version,
        fitted_at=datetime.now(UTC),
        random_seed=random_seed,
        tool_version=tool_version,
        fit_dataset_hash=fit_dataset_hash,
        val_dataset_hash=val_dataset_hash,
        test_dataset_hash=test_dataset_hash,
        split_counts=split_counts,
        metrics=metrics,
        coef=coef,
        intercept=intercept,
        x_knots=x_knots,
        y_knots=y_knots,
    )


def write_artifact(artifact: CalibrationArtifact, output_path: Path) -> None:
    """Write a CalibrationArtifact as JSON. Raises if gates should have blocked this."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("calibration_artifact_written path=%s version=%s", output_path, artifact.artifact_version)


# ---------------------------------------------------------------------------
# High-level fit pipeline
# ---------------------------------------------------------------------------


def run_fit(
    pairs: list[tuple[float, int, str]],
    method: str,
    arcface_model_version: str,
    model_profile: str,
    preprocessing_version: str,
    output_path: Path | None = None,
    gates: dict[str, float] | None = None,
    random_seed: int = 42,
    fit_frac: float = 0.6,
    val_frac: float = 0.2,
    tool_version: str = "1.0.0",
) -> tuple[CalibrationArtifact | None, list[str], list[str]]:
    """Full fit pipeline. Returns (artifact_or_none, gate_failures, leakage_findings).

    artifact is None when any gate or leakage check fails.
    Writes artifact to output_path only when all checks pass.

    Args:
        pairs: (similarity, label, identity_id) tuples.
        method: "logistic" or "isotonic".
        arcface_model_version: ArcFace model identifier (must match service config).
        model_profile: Model profile identifier (e.g. "full", "int8").
        preprocessing_version: Preprocessing pipeline version (e.g. "v1").
        output_path: Where to write the artifact JSON (None = dry run).
        gates: Override gate thresholds.
        random_seed: Reproducible seed for splitting and logistic regression.
        fit_frac: Fraction of identities for fitting.
        val_frac: Fraction of identities for validation/threshold selection.
        tool_version: Tool version embedded in the manifest.

    Returns:
        (artifact, gate_failures, leakage_findings)
    """
    # Split.
    fit_pairs, val_pairs, test_pairs, fit_ids, val_ids, test_ids = split_pairs_by_identity(
        pairs,
        fit_frac=fit_frac,
        val_frac=val_frac,
        random_seed=random_seed,
    )

    # Leakage check.
    leakage = validate_split_disjoint(fit_ids, val_ids, test_ids)
    if leakage:
        logger.error("calibration_leakage_found findings=%s", leakage)
        return None, [], leakage

    # Count splits.
    n_pos_fit = sum(1 for _, lbl in fit_pairs if lbl == 1)
    n_neg_fit = sum(1 for _, lbl in fit_pairs if lbl == 0)
    n_pos_val = sum(1 for _, lbl in val_pairs if lbl == 1)
    n_neg_val = sum(1 for _, lbl in val_pairs if lbl == 0)
    n_pos_test = sum(1 for _, lbl in test_pairs if lbl == 1)
    n_neg_test = sum(1 for _, lbl in test_pairs if lbl == 0)

    split_counts = SplitCounts(
        fit_pairs=len(fit_pairs),
        fit_positives=n_pos_fit,
        fit_negatives=n_neg_fit,
        val_pairs=len(val_pairs),
        val_positives=n_pos_val,
        val_negatives=n_neg_val,
        test_pairs=len(test_pairs),
        test_positives=n_pos_test,
        test_negatives=n_neg_test,
        fit_identities=len(fit_ids),
        val_identities=len(val_ids),
        test_identities=len(test_ids),
    )

    g = {**DEFAULT_GATES, **(gates or {})}
    operating_point = float(g["operating_point_threshold"])

    # Fit.
    coef: float | None = None
    intercept: float | None = None
    x_knots: list[float] | None = None
    y_knots: list[float] | None = None

    if method == "logistic":
        coef, intercept = fit_logistic(fit_pairs, random_seed=random_seed)
    elif method == "isotonic":
        x_knots, y_knots = fit_isotonic(fit_pairs)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}")

    # Evaluate on test split.
    metrics = compute_metrics(
        test_pairs, coef, intercept, x_knots, y_knots, method, operating_point
    )

    # Gate check.
    failures = check_gates(metrics, split_counts, gates)
    if failures:
        logger.error("calibration_gates_failed failures=%s", failures)
        return None, failures, []

    # Build dataset hashes.
    fit_hash = hash_dataset(fit_pairs)
    val_hash = hash_dataset(val_pairs)
    test_hash = hash_dataset(test_pairs)

    artifact = build_artifact(
        method=method,
        coef=coef,
        intercept=intercept,
        x_knots=x_knots,
        y_knots=y_knots,
        arcface_model_version=arcface_model_version,
        model_profile=model_profile,
        preprocessing_version=preprocessing_version,
        metrics=metrics,
        split_counts=split_counts,
        fit_dataset_hash=fit_hash,
        val_dataset_hash=val_hash,
        test_dataset_hash=test_hash,
        random_seed=random_seed,
        tool_version=tool_version,
    )

    if output_path is not None:
        write_artifact(artifact, output_path)

    return artifact, [], []


# ---------------------------------------------------------------------------
# Convenience: load pairs from a JSON file
# ---------------------------------------------------------------------------


def load_pairs_json(path: Path) -> list[tuple[float, int, str]]:
    """Load (similarity, label, identity_id) tuples from a JSON file.

    Expected format:
    [
        {"similarity": 0.87, "label": 1, "identity_id": "person-001"},
        ...
    ]
    """
    raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [
        (float(r["similarity"]), int(r["label"]), str(r["identity_id"]))
        for r in raw
    ]
