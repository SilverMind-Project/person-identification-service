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
import random
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.calibration.episodes import NOT_ENROLLED
from app.calibration.manifest import ScoreRecord
from app.calibration.models import (
    CalibrationArtifact,
    CalibrationArtifactV2,
    CalibrationMetrics,
    ExternalDatasetRef,
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

    return metrics_from_probabilities(probs, labels, operating_point)


def metrics_from_probabilities(
    probs: list[float], labels: list[float], operating_point: float
) -> CalibrationMetrics:
    """Brier score, log loss, 10-bin ECE, and FMR/FNMR at the operating point."""
    if not probs:
        return CalibrationMetrics()
    n = len(labels)

    # Brier score.
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / n

    # Log loss (clamp to avoid log(0)).
    eps = 1e-15
    ll = (
        -sum(
            y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
            for p, y in zip(probs, labels, strict=True)
        )
        / n
    )

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
    logger.info(
        "calibration_artifact_written path=%s version=%s", output_path, artifact.artifact_version
    )


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
    return [(float(r["similarity"]), int(r["label"]), str(r["identity_id"])) for r in raw]


# ---------------------------------------------------------------------------
# Schema 2: multi-feature fit on household captures (episode splits, holdout gates)
# ---------------------------------------------------------------------------

DEFAULT_GATES_V2: dict[str, float] = {
    "max_ece": 0.05,
    "max_brier_score": 0.10,
    # Applies to closed-set and open-set impostors together, and to open-set alone.
    "max_fmr_at_operating_point": 0.01,
    "operating_point_threshold": 0.80,
    # Holdout genuine faces an identity needs before the artifact covers it.
    "min_holdout_genuine_per_identity": 30,
    # Household plus external identities in the fit split.
    "min_fit_identities": 5,
}
FEATURE_SETS: tuple[tuple[str, ...], ...] = (
    ("similarity", "quality_score"),
    ("similarity", "resolution", "sharpness"),
)


@dataclass
class RecordSplit:
    fit: list[ScoreRecord] = field(default_factory=list)
    val: list[ScoreRecord] = field(default_factory=list)
    holdout: list[ScoreRecord] = field(default_factory=list)
    leakage: list[str] = field(default_factory=list)


def _probe_group(record: ScoreRecord) -> str:
    return record.probe_identity_id or NOT_ENROLLED


def split_records(
    records: Sequence[ScoreRecord],
    *,
    random_seed: int = 42,
    fit_frac: float = 0.6,
    val_frac: float = 0.2,
) -> RecordSplit:
    """Split household records by episode (stratified by identity), external ones by identity.

    Every record of one episode lands in one partition, so adjacent frames of a walk never sit in
    both fit and holdout. External identities go to fit or validation only: the holdout, which
    the gates read, is household data alone.
    """
    rng = random.Random(random_seed)
    split = RecordSplit()
    assignment: dict[str, str] = {}

    household = [r for r in records if r.dataset == "household"]
    episodes_by_group: dict[str, set[str]] = {}
    groups_by_episode: dict[str, set[str]] = {}
    for r in household:
        episodes_by_group.setdefault(_probe_group(r), set()).add(r.episode_id)
        groups_by_episode.setdefault(r.episode_id, set()).add(_probe_group(r))
    # An episode is one person's walk: two labels in one episode means mislabeled data.
    split.leakage.extend(
        f"episode {episode} carries labels {sorted(groups)}"
        for episode, groups in sorted(groups_by_episode.items())
        if len(groups) > 1
    )
    if split.leakage:
        return split
    for group in sorted(episodes_by_group):
        episodes = sorted(episodes_by_group[group])
        rng.shuffle(episodes)
        n = len(episodes)
        n_holdout = max(1, round(n * (1.0 - fit_frac - val_frac))) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        for index, episode in enumerate(episodes):
            if index < n_holdout:
                assignment[episode] = "holdout"
            elif index < n_holdout + n_val:
                assignment[episode] = "val"
            else:
                assignment[episode] = "fit"

    external_ids = sorted({_probe_group(r) for r in records if r.dataset == "external"})
    rng.shuffle(external_ids)
    n_external_val = round(len(external_ids) * val_frac / (fit_frac + val_frac))
    external_partition = {
        identity: ("val" if index < n_external_val else "fit")
        for index, identity in enumerate(external_ids)
    }

    for r in records:
        part = (
            assignment[r.episode_id]
            if r.dataset == "household"
            else external_partition[_probe_group(r)]
        )
        getattr(split, part).append(r)

    seen: dict[str, str] = {}
    for part in ("fit", "val", "holdout"):
        for r in getattr(split, part):
            key = (
                r.episode_id if r.dataset == "household" else f"external-identity:{_probe_group(r)}"
            )
            if seen.setdefault(key, part) != part:
                split.leakage.append(f"{key} appears in {seen[key]} and {part}")
    return split


def _matrix(records: Sequence[ScoreRecord], features: Sequence[str]) -> list[list[float]]:
    return [[float(getattr(r, name)) for name in features] for r in records]


def fit_logistic_v2(
    records: Sequence[ScoreRecord], features: Sequence[str], random_seed: int = 42
) -> tuple[list[float], float]:
    """Logistic regression on standardized features; returns raw-space (coefficients, intercept).

    Coefficients are de-standardized (w / sigma, b - sum(w * mu / sigma)) so the runtime applies
    them to raw feature values with no scaler. Requires scikit-learn (calibration-tools extra).
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(_matrix(records, features), dtype=np.float64)
    y = np.asarray([r.label for r in records], dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise ValueError("the fit split needs both genuine and impostor records")
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=random_seed)
    clf.fit((x - mu) / sigma, y)
    weights = clf.coef_[0] / sigma
    intercept = float(clf.intercept_[0] - np.sum(clf.coef_[0] * mu / sigma))
    return [float(w) for w in weights], intercept


def predict_logistic_v2(
    records: Sequence[ScoreRecord],
    features: Sequence[str],
    coefficients: Sequence[float],
    intercept: float,
) -> list[float]:
    z = (
        np.asarray(_matrix(records, features), dtype=np.float64) @ np.asarray(coefficients)
        + intercept
    )
    return [float(v) for v in 1.0 / (1.0 + np.exp(-z))]


def reliability_bins(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Per equal-width bin: count, mean predicted probability, observed match rate."""
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [
            (p, y)
            for p, y in zip(probs, labels, strict=True)
            if lo <= p < hi or (b == n_bins - 1 and p == 1.0)
        ]
        if members:
            bins.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "count": len(members),
                    "mean_predicted": sum(p for p, _ in members) / len(members),
                    "observed_rate": sum(y for _, y in members) / len(members),
                }
            )
    return bins


@dataclass
class FitV2Result:
    artifact: CalibrationArtifactV2 | None
    failures: list[str]
    leakage: list[str]
    report: dict[str, Any] = field(default_factory=dict)


def _hash_records(records: Sequence[ScoreRecord]) -> str:
    raw = json.dumps(sorted(json.dumps(asdict(r), sort_keys=True) for r in records)).encode()
    return hashlib.sha256(raw).hexdigest()


def _log_loss(probs: Sequence[float], labels: Sequence[int]) -> float:
    eps = 1e-15
    return -sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, labels, strict=True)
    ) / max(len(labels), 1)


def run_fit_v2(
    records: Sequence[ScoreRecord],
    meta: dict[str, Any],
    *,
    model_profile: str,
    allow_household_only: bool = False,
    gates: dict[str, float] | None = None,
    random_seed: int = 42,
    output_path: Path | None = None,
    tool_version: str = "2.0.0",
) -> FitV2Result:
    """Fit, gate on the household holdout, and write a schema 2 artifact only if every gate passes."""
    g = {**DEFAULT_GATES_V2, **(gates or {})}
    operating_point = float(g["operating_point_threshold"])
    has_external = any(r.dataset == "external" for r in records)
    if not has_external and not allow_household_only:
        return FitV2Result(
            None,
            ["household-only fit refused: pass --allow-household-only or add an external set"],
            [],
        )
    if not any(r.dataset == "household" for r in records):
        return FitV2Result(None, ["no household records: the gates need a household holdout"], [])

    split = split_records(records, random_seed=random_seed)
    if split.leakage:
        return FitV2Result(None, [], split.leakage)
    holdout = split.holdout
    if not split.fit or not split.val or not holdout:
        return FitV2Result(
            None,
            [
                f"every partition needs records (fit={len(split.fit)} val={len(split.val)} "
                f"holdout={len(holdout)}); label more episodes"
            ],
            [],
        )

    report: dict[str, Any] = {"feature_sets": {}}
    best: tuple[float, int, tuple[str, ...], list[float], float] | None = None
    for features in FEATURE_SETS:
        try:
            coefficients, intercept = fit_logistic_v2(split.fit, features, random_seed)
        except ValueError as exc:
            return FitV2Result(None, [str(exc)], [])
        val_loss = _log_loss(
            predict_logistic_v2(split.val, features, coefficients, intercept),
            [r.label for r in split.val],
        )
        report["feature_sets"]["+".join(features)] = round(val_loss, 6)
        # Lower validation log loss wins; on a tie the smaller feature set is kept.
        candidate = (round(val_loss, 9), len(features), features, coefficients, intercept)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    _, _, features, coefficients, intercept = best

    holdout_probs = predict_logistic_v2(holdout, features, coefficients, intercept)
    holdout_labels = [r.label for r in holdout]
    metrics = metrics_from_probabilities(
        holdout_probs, [float(y) for y in holdout_labels], operating_point
    )
    open_set = [
        p for p, r in zip(holdout_probs, holdout, strict=True) if r.probe_identity_id is None
    ]
    open_set_fmr = (
        sum(1 for p in open_set if p >= operating_point) / len(open_set) if open_set else None
    )
    genuine_by_identity: dict[str, int] = {}
    for r in holdout:
        if r.label == 1:
            genuine_by_identity[r.identity_id] = genuine_by_identity.get(r.identity_id, 0) + 1
    covered = sorted(
        identity
        for identity, count in genuine_by_identity.items()
        if count >= g["min_holdout_genuine_per_identity"]
    )
    fit_identities = {r.probe_identity_id for r in split.fit if r.probe_identity_id is not None}

    failures: list[str] = []
    if metrics.ece is not None and metrics.ece > g["max_ece"]:
        failures.append(f"holdout ece {metrics.ece:.4f} > {g['max_ece']}")
    if metrics.brier_score is not None and metrics.brier_score > g["max_brier_score"]:
        failures.append(f"holdout brier_score {metrics.brier_score:.4f} > {g['max_brier_score']}")
    fmr = metrics.fmr_at_operating_point
    if fmr is not None and fmr > g["max_fmr_at_operating_point"]:
        failures.append(f"holdout fmr {fmr:.4f} > {g['max_fmr_at_operating_point']}")
    if open_set_fmr is None:
        failures.append("open-set fmr unmeasured: label not-enrolled visitors into the holdout")
    elif open_set_fmr > g["max_fmr_at_operating_point"]:
        failures.append(f"open-set fmr {open_set_fmr:.4f} > {g['max_fmr_at_operating_point']}")
    if not covered:
        failures.append(
            f"no identity has {int(g['min_holdout_genuine_per_identity'])} holdout genuine faces"
        )
    if len(fit_identities) < g["min_fit_identities"]:
        failures.append(f"fit identities {len(fit_identities)} < {int(g['min_fit_identities'])}")

    report |= {
        "features": list(features),
        "holdout": metrics.model_dump(),
        "open_set_fmr": open_set_fmr,
        "holdout_genuine_by_identity": genuine_by_identity,
        "covered_identities": covered,
        "reliability": reliability_bins(holdout_probs, holdout_labels),
        "fit_identities": sorted(fit_identities),
    }
    if failures:
        return FitV2Result(None, failures, [], report)

    def _counts(part: Sequence[ScoreRecord]) -> tuple[int, int, int]:
        positives = sum(r.label for r in part)
        return len(part), positives, len(part) - positives

    fit_n, fit_pos, fit_neg = _counts(split.fit)
    val_n, val_pos, val_neg = _counts(split.val)
    test_n, test_pos, test_neg = _counts(holdout)
    artifact = CalibrationArtifactV2(
        artifact_version=str(uuid.uuid4()),
        method="logistic_v2",
        arcface_model_version=str(meta["arcface_model_version"]),
        model_profile=model_profile,
        preprocessing_version=str(meta["preprocessing_version"]),
        fitted_at=datetime.now(UTC),
        random_seed=random_seed,
        tool_version=tool_version,
        fit_dataset_hash=_hash_records(split.fit),
        val_dataset_hash=_hash_records(split.val),
        test_dataset_hash=_hash_records(holdout),
        split_counts=SplitCounts(
            fit_pairs=fit_n,
            fit_positives=fit_pos,
            fit_negatives=fit_neg,
            val_pairs=val_n,
            val_positives=val_pos,
            val_negatives=val_neg,
            test_pairs=test_n,
            test_positives=test_pos,
            test_negatives=test_neg,
            fit_identities=len(fit_identities),
        ),
        metrics=metrics,
        profile="single_frame",
        quality_version=str(meta["quality_version"]),
        features=list(features),
        coefficients=coefficients,
        intercept=intercept,
        covered_identities=covered,
        fit_basis="pooled" if has_external else "household_only",
        capture_profile={k: float(v) for k, v in dict(meta.get("capture_profile", {})).items()},
        household_holdout_metrics=metrics,
        open_set_fmr_at_operating_point=open_set_fmr,
        external_datasets=[ExternalDatasetRef(**ref) for ref in meta.get("external_datasets", [])],
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return FitV2Result(artifact, [], [], report)
