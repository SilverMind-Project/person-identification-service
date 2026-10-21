"""Calibration capture: stratified daily caps, eligibility, the background store, and retention."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest

from app.services.calibration_candidates import InMemoryCalibrationCandidateRepository
from app.services.calibration_capture import (
    CalibrationCaptureService,
    CaptureContext,
    CapturePolicy,
)
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.face_quality import FaceQuality
from tests.calibration_fakes import VERSIONS, InMemoryCropStore

_T = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)
_KEYPOINTS = np.array(
    [[40.0, 50.0], [70.0, 50.0], [55.0, 65.0], [45.0, 80.0], [65.0, 80.0]], dtype=np.float32
)
_IMAGE = np.random.RandomState(0).randint(0, 255, (160, 160, 3), dtype=np.uint8)


def _face(quality_score: float = 0.45, det_score: float = 0.9, scored: bool = True) -> DetectedFace:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    face = DetectedFace(
        bbox=[20.0, 30.0, 90.0, 110.0],
        embedding=embedding,
        det_score=det_score,
        keypoints_5_px=_KEYPOINTS,
        yaw_deg=10.0,
        pitch_deg=-5.0,
    )
    if scored:
        face.quality = FaceQuality(
            score=quality_score,
            resolution=0.5,
            sharpness=0.6,
            detection=0.8,
            pitch=1.0,
            version="heuristic-v1",
        )
    return face


def _identity(best: str | None = "resident-alpha", similarity: float = 0.52) -> IdentifyResult:
    return IdentifyResult(
        person_id="unknown", name="Guest", bbox=[], best_candidate_id=best, similarity=similarity
    )


def _context(camera_id: str = "cam-1", at: datetime = _T) -> CaptureContext:
    return CaptureContext(
        camera_id=camera_id,
        captured_at=at,
        crop_pad_top_frac=0.15,
        crop_pad_side_frac=0.1,
        crop_jpeg_quality=95,
    )


def _service(
    repo: InMemoryCalibrationCandidateRepository | None = None,
    crops: InMemoryCropStore | None = None,
    **policy: object,
) -> CalibrationCaptureService:
    return CalibrationCaptureService(
        repo or InMemoryCalibrationCandidateRepository(),
        crops or InMemoryCropStore(),
        replace(CapturePolicy(enabled=True, per_bucket_per_day=2), **policy),  # type: ignore[arg-type]
        VERSIONS,
    )


def test_disabled_captures_nothing() -> None:
    service = _service(enabled=False)
    assert service.offer(_IMAGE, [_face()], [_identity()], _context()) == 0


def test_a_call_without_capture_context_captures_nothing() -> None:
    assert _service().offer(_IMAGE, [_face()], [_identity()], None) == 0


@pytest.mark.parametrize(
    "face",
    [_face(det_score=0.5), _face(scored=False), replace(_face(), keypoints_5_px=None)],
    ids=["low_detection", "unscored", "no_keypoints"],
)
def test_ineligible_faces_are_skipped(face: DetectedFace) -> None:
    assert _service().offer(_IMAGE, [face], [_identity()], _context()) == 0


def test_daily_cap_is_per_quality_decile_identity_and_camera() -> None:
    service = _service()
    same_bucket = [_face(0.41), _face(0.45), _face(0.49)]
    assert service.offer(_IMAGE, same_bucket, [_identity()] * 3, _context()) == 2
    # Another decile, another nearest identity, and another camera are separate buckets.
    assert service.offer(_IMAGE, [_face(0.95)], [_identity()], _context()) == 1
    assert service.offer(_IMAGE, [_face(0.45)], [_identity("resident-beta")], _context()) == 1
    assert service.offer(_IMAGE, [_face(0.45)], [_identity(None)], _context()) == 1
    assert service.offer(_IMAGE, [_face(0.45)], [_identity()], _context("cam-2")) == 1
    # The full bucket opens again the next UTC day.
    tomorrow = _context(at=_T + timedelta(days=1))
    assert service.offer(_IMAGE, [_face(0.45)], [_identity()], tomorrow) == 1


def test_a_full_queue_drops_rather_than_blocks() -> None:
    service = _service(queue_size=1, per_bucket_per_day=10)
    assert service.offer(_IMAGE, [_face(), _face()], [_identity()] * 2, _context()) == 1


async def test_started_service_stores_candidate_and_aligned_crop() -> None:
    repo, crops = InMemoryCalibrationCandidateRepository(), InMemoryCropStore()
    service = _service(repo, crops)
    await service.start()
    try:
        service.offer(_IMAGE, [_face(0.45)], [_identity()], _context())
        await service.drain()
    finally:
        await service.stop()

    [candidate] = (await repo.list_page(state="unlabeled", limit=10, cursor=None)).candidates
    assert (candidate.camera_id, candidate.captured_at, candidate.source) == (
        "cam-1",
        _T,
        "cts_identify",
    )
    assert (candidate.quality_score, candidate.quality_version) == (0.45, "heuristic-v1")
    assert candidate.quality_components == {
        "resolution": 0.5,
        "sharpness": 0.6,
        "detection": 0.8,
        "pitch": 1.0,
    }
    assert candidate.capture_profile == {
        "crop_pad_top_frac": 0.15,
        "crop_pad_side_frac": 0.1,
        "crop_jpeg_quality": 95,
    }
    assert (candidate.nearest_person_id, candidate.nearest_similarity) == ("resident-alpha", 0.52)
    assert (candidate.arcface_model_version, candidate.interocular_px) == ("buffalo_l", 30.0)
    crop = cv2.imdecode(np.frombuffer(crops.objects[candidate.crop_object_key], np.uint8), 1)
    assert crop.shape == (112, 112, 3)


async def test_start_seeds_todays_caps_from_stored_captures() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    first = _service(repo)
    await first.start()
    first.offer(
        _IMAGE, [_face(0.45), _face(0.46)], [_identity()] * 2, _context(at=datetime.now(UTC))
    )
    await first.drain()
    await first.stop()

    restarted = _service(repo)
    await restarted.start()
    try:
        assert (
            restarted.offer(_IMAGE, [_face(0.45)], [_identity()], _context(at=datetime.now(UTC)))
            == 0
        )
    finally:
        await restarted.stop()


class _FailingRepo(InMemoryCalibrationCandidateRepository):
    async def insert(self, candidate: object) -> None:  # type: ignore[override]
        raise ConnectionError("database unavailable")


async def test_failed_insert_removes_the_uploaded_crop() -> None:
    crops = InMemoryCropStore()
    service = _service(_FailingRepo(), crops)
    await service.start()
    try:
        service.offer(_IMAGE, [_face()], [_identity()], _context())
        await service.drain()
    finally:
        await service.stop()
    assert crops.objects == {}


async def test_purge_and_delete_remove_crops() -> None:
    repo, crops = InMemoryCalibrationCandidateRepository(), InMemoryCropStore()
    service = _service(repo, crops, per_bucket_per_day=10)
    await service.start()
    old = _context(at=_T - timedelta(days=40))
    service.offer(_IMAGE, [_face(), _face(), _face()], [_identity()] * 3, old)
    await service.drain()
    await service.stop()
    labeled, rejected, unlabeled = (
        await repo.list_page(state=None, limit=10, cursor=None)
    ).candidates
    await repo.label(
        labeled.candidate_id,
        kind="not_enrolled",
        identity_id=None,
        source="operator_reviewed",
        labeled_by="op",
        labeled_at=_T,
    )
    await repo.reject(rejected.candidate_id)

    assert await service.purge_expired(_T) == 2
    assert set(crops.objects) == {labeled.crop_object_key}
    assert await service.delete_candidate(labeled.candidate_id) is True
    assert await service.delete_candidate(labeled.candidate_id) is False
    assert crops.objects == {}
    assert unlabeled.candidate_id not in {c.candidate_id for c in await repo.list_labeled()}
