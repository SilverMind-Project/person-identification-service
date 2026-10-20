"""Offline analysis behind ``scripts/calibrate_face_quality.py``: suggest face quality ramp anchors.

Given the four raw quality inputs of labeled faces and each face's raw similarity to its labeled
identity, report each input's distribution, its Spearman rank correlation with raw similarity
(does the input predict recognizability at all?), and suggested ramp anchors at the 5th and 95th
percentiles. The output is advice for a human editing ``config/settings.yaml``; nothing here
writes configuration. Requires the ``calibration-tools`` extra (scipy); never imported by the
service.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass(frozen=True)
class FaceQualitySample:
    """Raw quality inputs of one labeled face and its similarity to the labeled identity."""

    interocular_px: float | None
    det_score: float
    abs_pitch_deg: float
    log10_laplacian_variance: float
    raw_similarity: float


@dataclass(frozen=True)
class InputReport:
    """Distribution, predictive power, and suggested ramp anchors for one quality input."""

    name: str
    count: int
    p5: float
    p50: float
    p95: float
    # Spearman rank correlation with raw similarity; None with fewer than three faces or a
    # constant input.
    spearman_rho: float | None
    spearman_p_value: float | None
    # The ramp's 0 and 1 anchors. For pitch (a larger angle is worse) the low anchor is the 95th
    # percentile and the high anchor the 5th.
    suggested_low: float
    suggested_high: float


# (input name, accessor, larger is better)
_INPUTS = (
    ("interocular_px", lambda s: s.interocular_px, True),
    ("det_score", lambda s: s.det_score, True),
    ("log10_laplacian_variance", lambda s: s.log10_laplacian_variance, True),
    ("abs_pitch_deg", lambda s: s.abs_pitch_deg, False),
)


def _report(
    name: str, values: Sequence[float], similarities: Sequence[float], larger_is_better: bool
) -> InputReport:
    array = np.asarray(values, dtype=np.float64)
    p5, p50, p95 = (float(v) for v in np.percentile(array, [5, 50, 95]))
    rho: float | None = None
    p_value: float | None = None
    if len(array) >= 3 and float(np.ptp(array)) > 0.0 and float(np.ptp(similarities)) > 0.0:
        result = spearmanr(array, np.asarray(similarities, dtype=np.float64))
        rho = float(result.statistic)
        p_value = float(result.pvalue)
        if math.isnan(rho):
            rho = p_value = None
    low, high = (p5, p95) if larger_is_better else (p95, p5)
    return InputReport(
        name=name,
        count=len(array),
        p5=p5,
        p50=p50,
        p95=p95,
        spearman_rho=rho,
        spearman_p_value=p_value,
        suggested_low=low,
        suggested_high=high,
    )


def analyze(samples: Sequence[FaceQualitySample]) -> list[InputReport]:
    """Report every quality input; faces with an unmeasurable interocular distance are skipped
    for that input only."""
    if not samples:
        raise ValueError("no labeled faces to analyze")
    reports: list[InputReport] = []
    for name, accessor, larger_is_better in _INPUTS:
        pairs = [(accessor(s), s.raw_similarity) for s in samples if accessor(s) is not None]
        if not pairs:
            continue
        values, similarities = zip(*pairs, strict=True)
        reports.append(_report(name, values, similarities, larger_is_better))
    return reports
