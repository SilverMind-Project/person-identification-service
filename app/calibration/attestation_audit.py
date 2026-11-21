"""Audit arithmetic for wearable-attested face labels. Pure, stdlib only.

An attested label says "a wearable placed this member on the track this face came from". Before
such labels may calibrate anything, an operator audits a uniform random sample of them; the
labels qualify only when at least ``MIN_AUDITED`` were audited and the Wilson 95% lower bound of
their precision is at least ``MIN_PRECISION_LOWER_BOUND``. The bar is fixed here, not in
settings, so no configuration can lower it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.calibration_candidates import CalibrationCandidate

MIN_AUDITED = 200
MIN_PRECISION_LOWER_BOUND = 0.99
_Z_95 = 1.959963984540054


def wilson_lower_bound(successes: int, trials: int, z: float = _Z_95) -> float | None:
    """Lower end of the Wilson score interval for a binomial proportion; None without trials."""
    if trials <= 0:
        return None
    if not 0 <= successes <= trials:
        raise ValueError("successes must be within [0, trials]")
    p = successes / trials
    z2 = z * z
    centre = p + z2 / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
    return max(0.0, (centre - margin) / (1 + z2 / trials))


@dataclass(frozen=True)
class AuditSummary:
    attested: int
    audited: int
    correct: int
    wrong: int
    not_a_face: int
    # Wilson 95% lower bound of correct / audited; None before any audit.
    precision_lower_bound: float | None

    @property
    def meets_bar(self) -> bool:
        return (
            self.audited >= MIN_AUDITED
            and self.precision_lower_bound is not None
            and self.precision_lower_bound >= MIN_PRECISION_LOWER_BOUND
        )


def summarize(attested: Sequence[CalibrationCandidate]) -> AuditSummary:
    verdicts = [c.audit_verdict for c in attested if c.audit_verdict is not None]
    correct = sum(v == "correct" for v in verdicts)
    return AuditSummary(
        attested=len(attested),
        audited=len(verdicts),
        correct=correct,
        wrong=sum(v == "wrong" for v in verdicts),
        not_a_face=sum(v == "not_a_face" for v in verdicts),
        precision_lower_bound=wilson_lower_bound(correct, len(verdicts)),
    )


def audit_sample(
    unaudited: Sequence[CalibrationCandidate], n: int, *, seed: int | None = None
) -> list[CalibrationCandidate]:
    """A uniform random sample without replacement; the auditor must not choose the faces."""
    rng = random.Random(seed)
    return rng.sample(list(unaudited), k=min(n, len(unaudited)))
