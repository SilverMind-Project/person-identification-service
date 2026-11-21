"""Offline recognizer comparison and adoption-gate evaluator.

Compares a candidate recognizer profile against the baseline recognizer (for example ``buffalo_l``)
on a labeled household holdout split:

- Operating point: similarity threshold achieving at most 1% open-set false match rate (FMR) on
  held-out ``not_enrolled`` probes.
- Stratified genuine acceptance rate (GAR) at that 1% open-set FMR operating point, reported
  across all four household resolution bands (``lt_16px``, ``16_to_24px``, ``24_to_40px``,
  ``ge_40px``), the low-resolution gate band (``16_to_32px``), usable faces (``ge_24px``), and
  ``all`` covered enrolled residents.
- Expected calibration error (ECE) on the household holdout for calibrated probabilities.
- Explicit adoption gate: adopt only when the candidate improves genuine acceptance in the
  ``16_to_32px`` band by at least 10 percentage points (+0.10) at equal or better open-set FMR
  (<= baseline and <= 0.01) and acceptable holdout ECE (<= 0.05).

Pure stdlib + numpy + Pydantic (no scikit-learn dependency at import time) so the report schema
and evaluation math are deterministic and fast to test.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

REPORT_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_MAX_OPEN_SET_FMR = 0.01
DEFAULT_MIN_GAIN_16_TO_32PX_PP = 10.0
DEFAULT_MAX_ECE = 0.05

# Interocular bands in pixels: the four household resolution report bands, plus the 16-32 px
# adoption-gate band, the >=24 px usable-face band, and the full holdout ("all").
INTEROCULAR_BANDS: tuple[tuple[str, float | None, float | None], ...] = (
    ("lt_16px", None, 16.0),
    ("16_to_24px", 16.0, 24.0),
    ("24_to_40px", 24.0, 40.0),
    ("ge_40px", 40.0, None),
    ("16_to_32px", 16.0, 32.0),
    ("ge_24px", 24.0, None),
    ("all", None, None),
)


class RecognizerLicenceRecord(BaseModel):
    """Recorded licence terms across code, weights, and training dataset for one candidate."""

    profile_name: str
    recognizer_version: str
    embedding_dim: int = Field(..., ge=1)
    code_licence: str = Field(..., min_length=1)
    weights_licence: str = Field(..., min_length=1)
    training_data_licence: str = Field(..., min_length=1)
    commercial_clean: bool
    notes: str = ""


CANDIDATE_LICENCE_AUDIT: dict[str, RecognizerLicenceRecord] = {
    "buffalo_l": RecognizerLicenceRecord(
        profile_name="buffalo_l",
        recognizer_version="buffalo_l",
        embedding_dim=512,
        code_licence="MIT (InsightFace repository code)",
        weights_licence=(
            "InsightFace model zoo: ALL models are available for non-commercial research "
            "purposes only"
        ),
        training_data_licence=(
            "WebFace600K / MS1MV2-class academic face sets (non-commercial research only)"
        ),
        commercial_clean=False,
        notes="Current default recognizer. Disclosed in README and docs; deployers own licensing.",
    ),
    "adaface_ir50_webface4m": RecognizerLicenceRecord(
        profile_name="adaface_ir50_webface4m",
        recognizer_version="adaface_ir50_webface4m",
        embedding_dim=512,
        code_licence="MIT (mk-minchul/AdaFace)",
        weights_licence=(
            "Distributed with MIT codebase, but pretrained weights inherit training-data terms"
        ),
        training_data_licence=(
            "WebFace4M (non-commercial academic research only; inherits into pretrained weights)"
        ),
        commercial_clean=False,
        notes=(
            "Low-resolution quality-adaptive margin baseline (112x112 BGR input, 512-dim). "
            "Viable for non-commercial household evaluation; not commercially clean without "
            "retraining on consented/commercial data."
        ),
    ),
    "opencv_sface_2021dec": RecognizerLicenceRecord(
        profile_name="opencv_sface_2021dec",
        recognizer_version="opencv_sface_2021dec",
        embedding_dim=128,
        code_licence="Apache-2.0 (OpenCV Zoo)",
        weights_licence="Apache-2.0 (OpenCV Zoo face_recognition_sface_2021dec.onnx)",
        training_data_licence=(
            "Trained on academic face datasets (CASIA-WebFace/MS1M lineage); upstream data "
            "consent and commercial clearance remain unverified"
        ),
        commercial_clean=False,
        notes=(
            "128-dim embedding requires profile-scoped tables via a dedicated migration. "
            "Permissive weight licence file, but academic training-data provenance still requires "
            "legal review for commercial use."
        ),
    ),
    "dlib_resnet_v1": RecognizerLicenceRecord(
        profile_name="dlib_resnet_v1",
        recognizer_version="dlib_resnet_v1",
        embedding_dim=128,
        code_licence="BSL-1.0 (dlib)",
        weights_licence="Public domain / CC0 (author release)",
        training_data_licence=(
            "Curated mix including VGGFace, FaceScrub, and web-scraped faces (academic/unconsented "
            "biometric data provenance caveats)"
        ),
        commercial_clean=False,
        notes="128-dim legacy backbone; lower accuracy on low-resolution indoor faces.",
    ),
}


@dataclass(frozen=True)
class HoldoutProbeScore:
    """One labeled household holdout probe scored by one recognizer profile.

    ``is_open_set`` is True for ``not_enrolled`` probes (impostors outside the household roster);
    ``is_genuine`` is True when the probe is a covered enrolled resident scored against their own
    leave-episode-out gallery.
    """

    probe_id: str
    identity_id: str | None
    similarity: float
    calibrated_probability: float | None
    is_genuine: bool
    is_open_set: bool
    interocular_px: float | None


class BandEvalMetrics(BaseModel):
    """Per-interocular-band genuine acceptance and calibration error on covered residents."""

    band: str
    genuine_count: int = Field(..., ge=0)
    genuine_accepted: int = Field(..., ge=0)
    genuine_acceptance_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_genuine_similarity: float | None = None
    ece: float | None = Field(default=None, ge=0.0, le=1.0)


class ProfileEvalSummary(BaseModel):
    """Household holdout evaluation metrics for a single recognizer profile."""

    profile_name: str
    recognizer_version: str
    covered_identities: list[str]
    genuine_count: int = Field(..., ge=0)
    open_set_impostor_count: int = Field(..., ge=0)
    operating_threshold_at_1pct_fmr: float | None = None
    achieved_open_set_fmr: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_gar_at_1pct_fmr: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_ece: float | None = Field(default=None, ge=0.0, le=1.0)
    by_interocular_band: dict[str, BandEvalMetrics]
    dataset_hash: str


class AdoptionDecision(BaseModel):
    """Deterministic adopt / keep_baseline / insufficient_data outcome with reasons."""

    decision: Literal["adopt", "keep_baseline", "insufficient_data"]
    gain_16_to_32px_pp: float | None = None
    min_required_gain_pp: float = DEFAULT_MIN_GAIN_16_TO_32PX_PP
    open_set_fmr_ok: bool = False
    ece_ok: bool = False
    reasons: list[str] = Field(default_factory=list)


class RecognizerComparisonReport(BaseModel):
    """Schema-validated comparison report produced by Part B evaluation."""

    schema_version: Literal[1] = REPORT_SCHEMA_VERSION
    random_seed: int = 42
    max_open_set_fmr: float = DEFAULT_MAX_OPEN_SET_FMR
    min_required_gain_16_to_32px_pp: float = DEFAULT_MIN_GAIN_16_TO_32PX_PP
    max_ece: float = DEFAULT_MAX_ECE
    baseline: ProfileEvalSummary
    candidate: ProfileEvalSummary
    baseline_licence: RecognizerLicenceRecord
    candidate_licence: RecognizerLicenceRecord
    decision: AdoptionDecision


def _in_band(interocular_px: float | None, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is None:
        return True
    if interocular_px is None:
        return False
    if lo is not None and interocular_px < lo:
        return False
    return not (hi is not None and interocular_px >= hi)


def compute_ece(
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    num_bins: int = 10,
) -> float | None:
    """Expected calibration error over uniform probability bins in [0, 1]."""
    if not probabilities or len(probabilities) != len(labels):
        return None
    probs = np.asarray(probabilities, dtype=np.float64)
    ys = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    ece = 0.0
    for idx in range(num_bins):
        lo = idx / num_bins
        hi = (idx + 1) / num_bins
        if idx == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(probs[mask]))
        avg_acc = float(np.mean(ys[mask]))
        ece += (count / n) * abs(avg_acc - avg_conf)
    return round(float(ece), 6)


def threshold_for_open_set_fmr(
    open_set_similarities: Sequence[float],
    *,
    max_fmr: float = DEFAULT_MAX_OPEN_SET_FMR,
) -> tuple[float | None, float | None]:
    """Return the lowest similarity threshold achieving open-set FMR <= *max_fmr*.

    Checks exact empirical FMR (fraction of open-set scores ``>= threshold``) so the operating
    point is deterministic and conservative even on small household holdouts.
    """
    if not open_set_similarities:
        return None, None
    scores = np.sort(np.asarray(open_set_similarities, dtype=np.float64))
    n = len(scores)
    max_false_matches = int(np.floor(max_fmr * n + 1e-12))
    if max_false_matches >= n:
        return float(scores[0]), 1.0
    if max_false_matches == 0:
        # Strict threshold just above the largest impostor score so FMR is 0.0.
        threshold = float(np.nextafter(scores[-1], np.inf))
        return threshold, 0.0
    # Candidate threshold at the (n - max_false_matches)-th sorted score (0-indexed); if ties exist
    # that push FMR above max_fmr, step strictly above the tied value.
    idx = n - max_false_matches
    threshold = float(scores[idx])
    fmr = float(np.mean(scores >= threshold))
    if fmr > max_fmr + 1e-12:
        threshold = float(np.nextafter(threshold, np.inf))
        fmr = float(np.mean(scores >= threshold))
    return threshold, round(fmr, 6)


def _hash_probes(probes: Sequence[HoldoutProbeScore]) -> str:
    canonical = sorted(
        (
            p.probe_id,
            p.identity_id or "",
            round(p.similarity, 8),
            None if p.calibrated_probability is None else round(p.calibrated_probability, 8),
            p.is_genuine,
            p.is_open_set,
            None if p.interocular_px is None else round(p.interocular_px, 4),
        )
        for p in probes
    )
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_profile_holdout(
    *,
    profile_name: str,
    recognizer_version: str,
    probes: Sequence[HoldoutProbeScore],
    covered_identities: Sequence[str],
    max_open_set_fmr: float = DEFAULT_MAX_OPEN_SET_FMR,
) -> ProfileEvalSummary:
    """Summarize genuine acceptance at 1% open-set FMR by interocular band plus ECE."""
    covered_set = sorted(set(covered_identities))
    covered_lookup = set(covered_set)

    genuine_probes = [
        p
        for p in probes
        if p.is_genuine and p.identity_id is not None and p.identity_id in covered_lookup
    ]
    open_set_probes = [p for p in probes if p.is_open_set]

    threshold, achieved_fmr = threshold_for_open_set_fmr(
        [p.similarity for p in open_set_probes],
        max_fmr=max_open_set_fmr,
    )

    # Overall ECE across every scored probe with a calibrated probability (genuine + impostor).
    cal_probes = [
        p
        for p in probes
        if p.calibrated_probability is not None
        and (p.is_open_set or (p.identity_id is not None and p.identity_id in covered_lookup))
    ]
    overall_ece = compute_ece(
        [
            float(p.calibrated_probability)
            for p in cal_probes
            if p.calibrated_probability is not None
        ],
        [1 if p.is_genuine else 0 for p in cal_probes],
    )

    by_band: dict[str, BandEvalMetrics] = {}
    for band_name, lo, hi in INTEROCULAR_BANDS:
        band_genuine = [p for p in genuine_probes if _in_band(p.interocular_px, lo, hi)]
        count = len(band_genuine)
        if count == 0 or threshold is None:
            accepted = 0
            gar: float | None = None
            mean_sim: float | None = (
                round(float(np.mean([p.similarity for p in band_genuine])), 6) if count else None
            )
        else:
            accepted = sum(1 for p in band_genuine if p.similarity >= threshold)
            gar = round(accepted / count, 6)
            mean_sim = round(float(np.mean([p.similarity for p in band_genuine])), 6)

        band_cal = [
            p
            for p in cal_probes
            if _in_band(p.interocular_px, lo, hi) and p.calibrated_probability is not None
        ]
        band_ece = compute_ece(
            [
                float(p.calibrated_probability)
                for p in band_cal
                if p.calibrated_probability is not None
            ],
            [1 if p.is_genuine else 0 for p in band_cal],
        )
        by_band[band_name] = BandEvalMetrics(
            band=band_name,
            genuine_count=count,
            genuine_accepted=accepted,
            genuine_acceptance_rate=gar,
            mean_genuine_similarity=mean_sim,
            ece=band_ece,
        )

    return ProfileEvalSummary(
        profile_name=profile_name,
        recognizer_version=recognizer_version,
        covered_identities=covered_set,
        genuine_count=len(genuine_probes),
        open_set_impostor_count=len(open_set_probes),
        operating_threshold_at_1pct_fmr=None if threshold is None else round(threshold, 6),
        achieved_open_set_fmr=achieved_fmr,
        overall_gar_at_1pct_fmr=by_band["all"].genuine_acceptance_rate,
        overall_ece=overall_ece,
        by_interocular_band=by_band,
        dataset_hash=_hash_probes(probes),
    )


def evaluate_recognizer_comparison(
    *,
    baseline_profile: str,
    baseline_recognizer_version: str,
    baseline_probes: Sequence[HoldoutProbeScore],
    baseline_covered_identities: Sequence[str],
    candidate_profile: str,
    candidate_recognizer_version: str,
    candidate_probes: Sequence[HoldoutProbeScore],
    candidate_covered_identities: Sequence[str],
    baseline_licence: RecognizerLicenceRecord | None = None,
    candidate_licence: RecognizerLicenceRecord | None = None,
    random_seed: int = 42,
    max_open_set_fmr: float = DEFAULT_MAX_OPEN_SET_FMR,
    min_gain_16_to_32px_pp: float = DEFAULT_MIN_GAIN_16_TO_32PX_PP,
    max_ece: float = DEFAULT_MAX_ECE,
) -> RecognizerComparisonReport:
    """Compare baseline and candidate recognizer holdout scores and decide adoption."""
    base_summary = evaluate_profile_holdout(
        profile_name=baseline_profile,
        recognizer_version=baseline_recognizer_version,
        probes=baseline_probes,
        covered_identities=baseline_covered_identities,
        max_open_set_fmr=max_open_set_fmr,
    )
    cand_summary = evaluate_profile_holdout(
        profile_name=candidate_profile,
        recognizer_version=candidate_recognizer_version,
        probes=candidate_probes,
        covered_identities=candidate_covered_identities,
        max_open_set_fmr=max_open_set_fmr,
    )

    base_lic = baseline_licence or CANDIDATE_LICENCE_AUDIT.get(
        baseline_profile,
        RecognizerLicenceRecord(
            profile_name=baseline_profile,
            recognizer_version=baseline_recognizer_version,
            embedding_dim=512,
            code_licence="unspecified",
            weights_licence="unspecified",
            training_data_licence="unspecified",
            commercial_clean=False,
        ),
    )
    cand_lic = candidate_licence or CANDIDATE_LICENCE_AUDIT.get(
        candidate_profile,
        RecognizerLicenceRecord(
            profile_name=candidate_profile,
            recognizer_version=candidate_recognizer_version,
            embedding_dim=512,
            code_licence="unspecified",
            weights_licence="unspecified",
            training_data_licence="unspecified",
            commercial_clean=False,
        ),
    )

    reasons: list[str] = []
    base_16_32 = base_summary.by_interocular_band["16_to_32px"].genuine_acceptance_rate
    cand_16_32 = cand_summary.by_interocular_band["16_to_32px"].genuine_acceptance_rate

    if (
        base_summary.achieved_open_set_fmr is None
        or cand_summary.achieved_open_set_fmr is None
        or base_16_32 is None
        or cand_16_32 is None
    ):
        if base_summary.open_set_impostor_count == 0 or cand_summary.open_set_impostor_count == 0:
            reasons.append("open-set impostor holdout probes are missing")
        if base_16_32 is None or cand_16_32 is None:
            reasons.append("no genuine covered-resident probes in the 16_to_32px interocular band")
        decision = AdoptionDecision(
            decision="insufficient_data",
            gain_16_to_32px_pp=None,
            min_required_gain_pp=min_gain_16_to_32px_pp,
            open_set_fmr_ok=False,
            ece_ok=False,
            reasons=reasons,
        )
    else:
        gain_pp = round((cand_16_32 - base_16_32) * 100.0, 4)
        open_set_ok = (
            cand_summary.achieved_open_set_fmr <= max_open_set_fmr + 1e-12
            and cand_summary.achieved_open_set_fmr <= base_summary.achieved_open_set_fmr + 1e-12
        )
        ece_ok = (
            cand_summary.overall_ece is not None and cand_summary.overall_ece <= max_ece + 1e-12
        )
        gain_ok = gain_pp >= min_gain_16_to_32px_pp - 1e-9

        if not open_set_ok:
            reasons.append(
                f"candidate open-set FMR {cand_summary.achieved_open_set_fmr:.4f} exceeds "
                f"baseline {base_summary.achieved_open_set_fmr:.4f} or cap {max_open_set_fmr:.4f}"
            )
        if not gain_ok:
            reasons.append(
                f"16_to_32px genuine acceptance gain {gain_pp:+.2f} pp is below required "
                f"+{min_gain_16_to_32px_pp:.2f} pp"
            )
        if not ece_ok:
            reasons.append(
                f"candidate holdout ECE {cand_summary.overall_ece} exceeds cap {max_ece:.4f}"
            )
        if open_set_ok and gain_ok and ece_ok:
            reasons.append(
                f"candidate improves 16_to_32px GAR by {gain_pp:+.2f} pp at open-set FMR "
                f"{cand_summary.achieved_open_set_fmr:.4f} and ECE {cand_summary.overall_ece:.4f}"
            )
            verdict: Literal["adopt", "keep_baseline", "insufficient_data"] = "adopt"
        else:
            verdict = "keep_baseline"

        decision = AdoptionDecision(
            decision=verdict,
            gain_16_to_32px_pp=gain_pp,
            min_required_gain_pp=min_gain_16_to_32px_pp,
            open_set_fmr_ok=open_set_ok,
            ece_ok=ece_ok,
            reasons=reasons,
        )

    return RecognizerComparisonReport(
        schema_version=REPORT_SCHEMA_VERSION,
        random_seed=random_seed,
        max_open_set_fmr=max_open_set_fmr,
        min_required_gain_16_to_32px_pp=min_gain_16_to_32px_pp,
        max_ece=max_ece,
        baseline=base_summary,
        candidate=cand_summary,
        baseline_licence=base_lic,
        candidate_licence=cand_lic,
        decision=decision,
    )
