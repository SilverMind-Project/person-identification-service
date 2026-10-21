"""Fit v2: episode splits, raw-space coefficients, household-holdout gates, fail-closed coverage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression

from app.calibration.cli import main as cli_main
from app.calibration.evaluator import CalibrationEvaluator
from app.calibration.fit import predict_logistic_v2, run_fit_v2, split_records
from app.calibration.manifest import ScoreRecord

_META = {
    "arcface_model_version": "buffalo_l",
    "preprocessing_version": "v1",
    "quality_version": "heuristic-v1",
    "capture_profile": {
        "crop_pad_top_frac": 0.15,
        "crop_pad_side_frac": 0.1,
        "crop_jpeg_quality": 95,
    },
}
_STRONG = ["alpha", "beta", "gamma", "delta", "epsilon"]
_WEAK = "zeta"  # too few walks to reach the holdout genuine minimum


def _records(
    *,
    relatives: bool = False,
    open_set: bool = True,
    external: bool = False,
    seed: int = 3,
) -> list[ScoreRecord]:
    rng = np.random.RandomState(seed)
    gallery = [*_STRONG, _WEAK]
    records: list[ScoreRecord] = []

    def add(
        probe_id: str,
        episode: str,
        probe_identity: str | None,
        dataset: str = "household",
        relative_of: str | None = None,
    ) -> None:
        quality = float(rng.uniform(0.05, 0.95))
        targets = gallery if dataset == "household" else [f"external:{i}" for i in range(6)]
        for identity in targets:
            genuine = identity == probe_identity
            if genuine:
                similarity = rng.normal(0.38 + 0.3 * quality, 0.05)
            elif relative_of == identity:
                # A close relative of one resident scores higher than most genuine faces, which
                # no calibrator monotone in similarity can separate.
                similarity = rng.normal(0.85, 0.03)
            else:
                similarity = rng.normal(0.05, 0.05)
            records.append(
                ScoreRecord(
                    probe_id=probe_id,
                    episode_id=episode,
                    identity_id=identity,
                    probe_identity_id=probe_identity,
                    label=int(genuine),
                    similarity=float(similarity),
                    quality_score=quality,
                    resolution=quality,
                    sharpness=float(rng.uniform(0.2, 1.0)),
                    detection=0.9,
                    pitch=1.0,
                    camera_id="cam-1",
                    label_source="operator_reviewed"
                    if dataset == "household"
                    else "external_benchmark",
                    dataset=dataset,  # type: ignore[arg-type]
                )
            )

    for identity in _STRONG:
        for episode in range(20):
            for frame in range(8):
                add(f"{identity}-{episode}-{frame}", f"{identity}|cam-1|{episode}", identity)
    for episode in range(4):
        for frame in range(4):
            add(f"{_WEAK}-{episode}-{frame}", f"{_WEAK}|cam-1|{episode}", _WEAK)
    if open_set:
        for episode in range(15):
            for frame in range(8):
                add(f"visitor-{episode}-{frame}", f"not_enrolled|cam-1|{episode}", None)
    if relatives:
        for episode in range(15, 30):
            for frame in range(8):
                add(
                    f"relative-{episode}-{frame}",
                    f"not_enrolled|cam-1|{episode}",
                    None,
                    relative_of="alpha",
                )
    if external:
        for i in range(6):
            for frame in range(10):
                add(f"ext-{i}-{frame}", f"external|{i}", f"external:{i}", dataset="external")
    return records


def test_household_only_fit_passes_gates_and_covers_only_identities_with_enough_holdout() -> None:
    result = run_fit_v2(_records(), _META, model_profile="full", allow_household_only=True)

    assert result.failures == []
    artifact = result.artifact
    assert artifact is not None
    assert artifact.fit_basis == "household_only"
    assert artifact.covered_identities == sorted(_STRONG)
    assert artifact.quality_version == "heuristic-v1"
    assert artifact.capture_profile["crop_jpeg_quality"] == 95.0
    assert artifact.open_set_fmr_at_operating_point is not None
    assert artifact.open_set_fmr_at_operating_point <= 0.01
    assert artifact.features[0] == "similarity"
    assert result.report["reliability"]


def test_fit_is_deterministic() -> None:
    first = run_fit_v2(_records(), _META, model_profile="full", allow_household_only=True)
    second = run_fit_v2(_records(), _META, model_profile="full", allow_household_only=True)
    assert first.artifact is not None
    assert second.artifact is not None
    assert first.artifact.coefficients == second.artifact.coefficients
    assert first.artifact.intercept == second.artifact.intercept


def test_raw_space_coefficients_reproduce_sklearn_on_standardized_features() -> None:
    records = _records()
    result = run_fit_v2(records, _META, model_profile="full", allow_household_only=True)
    assert result.artifact is not None
    features = result.artifact.features
    split = split_records(records)
    x_fit = np.asarray([[getattr(r, f) for f in features] for r in split.fit])
    mu, sigma = x_fit.mean(axis=0), x_fit.std(axis=0)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit((x_fit - mu) / sigma, [r.label for r in split.fit])
    x_holdout = np.asarray([[getattr(r, f) for f in features] for r in split.holdout])
    expected = clf.predict_proba((x_holdout - mu) / sigma)[:, 1]
    actual = predict_logistic_v2(
        split.holdout,
        features,
        result.artifact.coefficients or [],
        result.artifact.intercept or 0.0,
    )
    assert np.allclose(actual, expected, atol=1e-6)


def test_household_only_fit_needs_the_explicit_flag() -> None:
    result = run_fit_v2(_records(), _META, model_profile="full")
    assert result.artifact is None
    assert "allow-household-only" in result.failures[0]


def test_external_set_makes_a_pooled_fit_and_never_reaches_the_holdout() -> None:
    records = _records(external=True)
    assert all(r.dataset == "household" for r in split_records(records).holdout)
    result = run_fit_v2(records, _META, model_profile="full")
    assert result.artifact is not None
    assert result.artifact.fit_basis == "pooled"


def test_open_set_fmr_gate_fails_on_lookalike_visitors() -> None:
    result = run_fit_v2(
        _records(relatives=True), _META, model_profile="full", allow_household_only=True
    )
    assert result.artifact is None
    assert any(f.startswith("open-set fmr") for f in result.failures)


def test_open_set_fmr_must_be_measured() -> None:
    result = run_fit_v2(
        _records(open_set=False), _META, model_profile="full", allow_household_only=True
    )
    assert result.artifact is None
    assert any("unmeasured" in f for f in result.failures)


def test_too_few_fit_identities_fails() -> None:
    result = run_fit_v2(
        _records(),
        _META,
        model_profile="full",
        allow_household_only=True,
        gates={"min_fit_identities": 9},
    )
    assert any(f.startswith("fit identities") for f in result.failures)


def test_an_episode_shared_by_two_labels_is_leakage() -> None:
    records = _records()
    # Re-label one beta probe into an alpha episode: one episode id now spans two groups.
    alpha_episode = next(r.episode_id for r in records if r.probe_identity_id == "alpha")
    tampered = [
        r
        if not r.probe_id.startswith("beta-0-")
        else type(r)(**{**r.__dict__, "episode_id": alpha_episode})
        for r in records
    ]
    result = run_fit_v2(tampered, _META, model_profile="full", allow_household_only=True)
    assert result.artifact is None
    assert result.leakage


def test_written_artifact_loads_and_fails_closed_outside_its_coverage(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    result = run_fit_v2(
        _records(), _META, model_profile="full", allow_household_only=True, output_path=output
    )
    assert result.artifact is not None
    assert json.loads(output.read_text())["schema_version"] == 2
    ev = CalibrationEvaluator.from_artifact_path(
        output,
        expected_arcface_model_version="buffalo_l",
        expected_model_profile="full",
        expected_preprocessing_version="v1",
        expected_quality_version="heuristic-v1",
    )
    assert ev.health() == "ready"
    features = {"similarity": 0.75, "quality_score": 0.8, "resolution": 0.8, "sharpness": 0.9}
    assert (ev.predict(features, "alpha") or 0.0) > 0.8
    assert ev.predict(features, _WEAK) is None
    assert ev.predict(features, "resident-enrolled-later") is None


def test_cli_validate_and_inspect_accept_a_fitted_v2_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "artifact.json"
    run_fit_v2(
        _records(), _META, model_profile="full", allow_household_only=True, output_path=output
    )

    assert (
        cli_main(["validate", "--artifact", str(output), "--quality-version", "heuristic-v1"]) == 0
    )
    printed = capsys.readouterr().out
    assert "PASS" in printed
    assert "covered_identities: alpha, beta, delta, epsilon, gamma" in printed
    assert cli_main(["inspect", "--artifact", str(output)]) == 0
    assert '"schema_version": 2' in capsys.readouterr().out
    assert cli_main(["validate", "--artifact", str(output), "--quality-version", "other-v2"]) == 1
