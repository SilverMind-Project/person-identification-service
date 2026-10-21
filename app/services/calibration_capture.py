"""Calibration capture: keep a stratified sample of live faces for an operator to label.

Calibration needs this household's own faces at every quality level, low quality especially, so
captures are capped per (quality decile, nearest identity, camera) per day. The decision is made
synchronously from in-memory counters; the crop upload and the insert run in a bounded background
worker, so ``/identify`` never waits on, or fails because of, a capture.

Captures happen only when ``calibration_capture.enabled`` is on and the request carries a capture
context (tracking-orchestrator sends one; other callers do not).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol

import cv2
import numpy as np

from app.observability.metrics import calibration_captures_total
from app.services.calibration_candidates import (
    NO_NEAREST,
    BucketKey,
    CalibrationCandidate,
    CalibrationCandidateRepository,
    quality_decile,
)
from app.services.face_engine import align_face
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.minio_client import MinioClient

logger = logging.getLogger(__name__)

CROP_PREFIX = "calibration"


@dataclass(frozen=True)
class CapturePolicy:
    enabled: bool = False
    # Captures kept per (quality decile, nearest identity or "none", camera) per UTC day.
    per_bucket_per_day: int = 40
    # Faces below the detector threshold are never kept (SCRFD already drops them; a guard).
    det_threshold: float = 0.6
    unlabeled_retention_days: int = 30
    # Pending captures waiting for upload; further faces are dropped (and counted) when full.
    queue_size: int = 64


@dataclass(frozen=True)
class CaptureContext:
    """Where and how the tracker produced the crop it sent."""

    camera_id: str
    captured_at: datetime
    crop_pad_top_frac: float
    crop_pad_side_frac: float
    crop_jpeg_quality: int


@dataclass(frozen=True)
class CaptureVersions:
    arcface_model_version: str
    preprocessing_version: str


class CropStore(Protocol):
    async def put_png(self, key: str, data: bytes) -> None: ...

    async def delete(self, keys: Sequence[str]) -> None: ...


class DailyBucketCounter:
    """Captures per sampling bucket for the current UTC day; resets when the day changes."""

    def __init__(self) -> None:
        self._day: date | None = None
        self._counts: dict[BucketKey, int] = {}

    def seed(self, day: date, counts: dict[BucketKey, int]) -> None:
        self._day, self._counts = day, dict(counts)

    def try_take(self, bucket: BucketKey, at: datetime, cap: int) -> bool:
        day = at.astimezone(UTC).date()
        if day != self._day:
            self._day, self._counts = day, {}
        if self._counts.get(bucket, 0) >= cap:
            return False
        self._counts[bucket] = self._counts.get(bucket, 0) + 1
        return True


def should_capture(face: DetectedFace, policy: CapturePolicy) -> str | None:
    """None when the face may be captured, else the reason it may not (a metric label)."""
    if face.quality is None:
        return "skipped_unscored"
    if face.keypoints_5_px is None:
        return "skipped_no_keypoints"
    if face.det_score < policy.det_threshold:
        return "skipped_low_detection"
    return None


@dataclass(frozen=True)
class _Pending:
    image: np.ndarray
    face: DetectedFace
    identity: IdentifyResult
    context: CaptureContext


class CalibrationCaptureService:
    def __init__(
        self,
        repository: CalibrationCandidateRepository,
        crops: CropStore,
        policy: CapturePolicy,
        versions: CaptureVersions,
    ) -> None:
        self._repository = repository
        self._crops = crops
        self._policy = policy
        self._versions = versions
        self._counter = DailyBucketCounter()
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue(maxsize=policy.queue_size)
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    async def start(self) -> None:
        """Seed today's caps from stored captures and start the upload worker."""
        if not self._policy.enabled:
            return
        today = datetime.now(UTC).date()
        start_of_day = datetime.combine(today, time.min, tzinfo=UTC)
        self._counter.seed(today, await self._repository.bucket_counts(start_of_day))
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    def offer(
        self,
        image: np.ndarray,
        faces: Sequence[DetectedFace],
        identities: Sequence[IdentifyResult],
        context: CaptureContext | None,
    ) -> int:
        """Queue the faces of one identify call that the policy keeps; returns how many."""
        if not self._policy.enabled or context is None:
            return 0
        queued = 0
        for face, identity in zip(faces, identities, strict=True):
            reason = should_capture(face, self._policy)
            if reason is None:
                assert face.quality is not None  # checked by should_capture
                bucket = (
                    quality_decile(face.quality.score),
                    identity.best_candidate_id or NO_NEAREST,
                    context.camera_id,
                )
                if not self._counter.try_take(
                    bucket, context.captured_at, self._policy.per_bucket_per_day
                ):
                    reason = "skipped_cap"
            if reason is None:
                try:
                    self._queue.put_nowait(_Pending(image, face, identity, context))
                except asyncio.QueueFull:
                    reason = "skipped_queue_full"
            if reason is None:
                queued += 1
                calibration_captures_total.labels(result="queued").inc()
            else:
                calibration_captures_total.labels(result=reason).inc()
        return queued

    async def drain(self) -> None:
        """Wait until every queued capture is stored or has failed (tests and shutdown)."""
        await self._queue.join()

    async def _run(self) -> None:
        while True:
            pending = await self._queue.get()
            try:
                await self._store(pending)
                calibration_captures_total.labels(result="stored").inc()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("calibration_capture_failed")
                calibration_captures_total.labels(result="failed").inc()
            finally:
                self._queue.task_done()

    async def _store(self, pending: _Pending) -> None:
        face, identity, context = pending.face, pending.identity, pending.context
        # Both are guaranteed by should_capture before a face is queued.
        assert face.quality is not None
        assert face.keypoints_5_px is not None
        candidate_id = str(uuid.uuid4())
        key = f"{CROP_PREFIX}/{context.captured_at.astimezone(UTC):%Y-%m-%d}/{candidate_id}.png"
        aligned = align_face(pending.image, face.keypoints_5_px)
        ok, png = cv2.imencode(".png", aligned)
        if not ok:
            raise RuntimeError("could not encode aligned crop")
        await self._crops.put_png(key, png.tobytes())
        quality = face.quality
        candidate = CalibrationCandidate(
            candidate_id=candidate_id,
            captured_at=context.captured_at,
            camera_id=context.camera_id,
            source="cts_identify",
            crop_object_key=key,
            embedding=np.asarray(face.embedding, dtype=np.float32),
            arcface_model_version=self._versions.arcface_model_version,
            preprocessing_version=self._versions.preprocessing_version,
            quality_version=quality.version,
            quality_score=quality.score,
            quality_components={
                "resolution": quality.resolution,
                "sharpness": quality.sharpness,
                "detection": quality.detection,
                "pitch": quality.pitch,
            },
            capture_profile={
                "crop_pad_top_frac": context.crop_pad_top_frac,
                "crop_pad_side_frac": context.crop_pad_side_frac,
                "crop_jpeg_quality": context.crop_jpeg_quality,
            },
            interocular_px=face.geometry.interocular_px,
            yaw_deg=face.yaw_deg,
            pitch_deg=face.pitch_deg,
            nearest_person_id=identity.best_candidate_id,
            nearest_similarity=identity.similarity,
        )
        try:
            await self._repository.insert(candidate)
        except Exception:
            # Never leave an orphan crop for a row that does not exist.
            await self._crops.delete([key])
            raise

    async def purge_expired(self, now: datetime) -> int:
        """Delete unlabeled and rejected captures past retention, with their crops."""
        cutoff = now - timedelta(days=self._policy.unlabeled_retention_days)
        keys = await self._repository.purge(cutoff)
        await self._crops.delete(keys)
        return len(keys)

    async def delete_candidate(self, candidate_id: str) -> bool:
        key = await self._repository.delete(candidate_id)
        if key is None:
            return False
        await self._crops.delete([key])
        return True


class MinioCropStore:
    """``CropStore`` over the service's MinIO client (boto3 calls run in a worker thread)."""

    def __init__(self, client: MinioClient) -> None:
        self._client = client

    async def put_png(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._client.upload_bytes, data, key, "image/png")

    async def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            await asyncio.to_thread(self._client.delete_object, key)
