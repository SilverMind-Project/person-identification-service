"""M00 characterization for the uncalibrated ArcFace confidence contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.identification import FaceDetection

_FIXTURE = Path(__file__).parent / "fixtures/identity_integrity/missing_calibration_artifact.json"


@pytest.mark.xfail(
    strict=True,
    reason="M10 removes this xfail when missing calibration fails closed in the API contract",
)
def test_missing_calibration_exposes_null_calibrated_confidence() -> None:
    data = json.loads(_FIXTURE.read_text())
    detection = FaceDetection(
        person_id="resident-alpha",
        name="Synthetic Resident Alpha",
        confidence=data["current_baseline"]["confidence"],
        similarity=data["raw_similarity"],
        recognition_state=data["recognition_state"],
        bbox=[10.0, 20.0, 110.0, 220.0],
    ).model_dump()

    assert detection["raw_similarity"] == data["raw_similarity"]
    assert detection["calibrated_confidence"] is None
    assert detection["calibration_status"] == data["target"]["calibration_status"]
