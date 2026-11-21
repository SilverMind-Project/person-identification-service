"""In-memory stand-ins for calibration capture in router and service tests."""

from __future__ import annotations

from collections.abc import Sequence

from app.services.calibration_candidates import InMemoryCalibrationCandidateRepository
from app.services.calibration_capture import (
    CalibrationCaptureService,
    CapturePolicy,
    CaptureVersions,
)

VERSIONS = CaptureVersions(recognizer_version="buffalo_l", preprocessing_version="v1")


class InMemoryCropStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_png(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.objects.pop(key, None)

    async def get_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def delete_prefix(self, prefix: str) -> list[str]:
        removed = sorted(k for k in self.objects if k.startswith(prefix))
        for k in removed:
            del self.objects[k]
        return removed


def disabled_capture() -> CalibrationCaptureService:
    """A capture service that never captures, for routers that do not test capture."""
    return CalibrationCaptureService(
        InMemoryCalibrationCandidateRepository(), InMemoryCropStore(), CapturePolicy(), VERSIONS
    )
