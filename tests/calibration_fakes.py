"""In-memory stand-ins for calibration capture in router and service tests."""

from __future__ import annotations

from collections.abc import Sequence

from app.services.calibration_candidates import InMemoryCalibrationCandidateRepository
from app.services.calibration_capture import (
    CalibrationCaptureService,
    CapturePolicy,
    CaptureVersions,
)

VERSIONS = CaptureVersions(arcface_model_version="buffalo_l", preprocessing_version="v1")


class InMemoryCropStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_png(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.objects.pop(key, None)


def disabled_capture() -> CalibrationCaptureService:
    """A capture service that never captures, for routers that do not test capture."""
    return CalibrationCaptureService(
        InMemoryCalibrationCandidateRepository(), InMemoryCropStore(), CapturePolicy(), VERSIONS
    )
