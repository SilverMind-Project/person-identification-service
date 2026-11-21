"""Calibration capture: keep a stratified sample of live faces for an operator to label.

Calibration needs this household's own faces at every quality level, low quality especially, so
captures are capped per (quality decile, nearest identity, camera) per day. The decision is made
synchronously from in-memory counters; the crop upload and the insert run in a bounded background
worker, so ``/identify`` never waits on, or fails because of, a capture.

Captures happen only when ``calibration_capture.enabled`` is on and the request carries a capture
context (tracking-orchestrator sends one; other callers do not).

Wearable attestation is a second, independent path through the same storage. With
``calibration_capture.attestation_enabled``, every scored face of an identify call whose capture
context names a detection is held briefly in memory (``AttestationCaptureBuffer``; nothing is
written). When tracking-orchestrator later reports that a wearable association placed a household
member on that detection, :meth:`CalibrationCaptureService.attest` stores the held faces as
``attested`` candidates labeled ``sensor_attested`` for that member. They are machine labels: an
operator audits a random sample, and only an operator promotes one into recognition.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol

import cv2
import numpy as np

from app.observability.metrics import attestation_faces_total, calibration_captures_total
from app.services.calibration_candidates import (
    NO_NEAREST,
    AttestationGrade,
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
    # The tracker's detection id for this crop; a face can be attested only through it.
    detection_id: str | None = None


@dataclass(frozen=True)
class AttestationPolicy:
    """The in-memory hold for wearable attestation and its admission floors."""

    enabled: bool = False
    # Faces held at once; the oldest goes first when full.
    max_items: int = 2000
    # A held face older than this (seconds, capture time) can no longer be attested.
    max_age_s: float = 600.0
    # heuristic-v1 face quality floor; faces without a score are never held.
    min_quality: float = 0.35
    # Faces smaller than this between the eyes are not usable exemplars or calibration probes.
    min_interocular_px: float = 24.0
    # Attested faces stored per (quality decile, attested identity, camera) per UTC day, so a
    # resident wearing a tag all day cannot swamp the set.
    per_bucket_per_day: int = 40


@dataclass(frozen=True)
class AttestationOutcome:
    moved: int
    missed: int
    capped: int


@dataclass(frozen=True)
class CaptureVersions:
    arcface_model_version: str = ""
    preprocessing_version: str = ""
    recognizer_version: str = ""

    def __post_init__(self) -> None:
        if self.recognizer_version and not self.arcface_model_version:
            object.__setattr__(self, "arcface_model_version", self.recognizer_version)
        elif self.arcface_model_version and not self.recognizer_version:
            object.__setattr__(self, "recognizer_version", self.arcface_model_version)


class CropStore(Protocol):
    async def put_png(self, key: str, data: bytes) -> None: ...

    async def delete(self, keys: Sequence[str]) -> None: ...

    async def get_bytes(self, key: str) -> bytes | None: ...

    async def delete_prefix(self, prefix: str) -> list[str]: ...


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


@dataclass(frozen=True)
class _Held:
    aligned: np.ndarray
    face: DetectedFace
    identity: IdentifyResult
    context: CaptureContext


def attestation_hold_rejection(face: DetectedFace, policy: AttestationPolicy) -> str | None:
    """None when the face may be held for attestation, else the reason (a metric label)."""
    if face.quality is None or face.keypoints_5_px is None:
        return "skipped_unscored"
    if face.quality.score < policy.min_quality:
        return "skipped_quality"
    interocular = face.geometry.interocular_px
    if interocular is None or interocular < policy.min_interocular_px:
        return "skipped_small"
    return None


class AttestationCaptureBuffer:
    """Aligned faces keyed by detection id, bounded by count and capture-time age.

    Biometric data held only in memory; it is written nowhere unless a face is attested.
    """

    def __init__(self, policy: AttestationPolicy) -> None:
        self._policy = policy
        self._items: OrderedDict[str, _Held] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def hold(self, detection_id: str, held: _Held) -> None:
        self._items.pop(detection_id, None)
        self._items[detection_id] = held
        while len(self._items) > self._policy.max_items:
            self._items.popitem(last=False)
            attestation_faces_total.labels(result="evicted").inc()

    def take(self, detection_id: str, *, now: datetime) -> _Held | None:
        held = self._items.pop(detection_id, None)
        if held is None:
            return None
        if now - held.context.captured_at > timedelta(seconds=self._policy.max_age_s):
            return None
        return held

    def expire(self, now: datetime) -> int:
        cutoff = now - timedelta(seconds=self._policy.max_age_s)
        stale = [k for k, h in self._items.items() if h.context.captured_at < cutoff]
        for key in stale:
            del self._items[key]
        return len(stale)


class CalibrationCaptureService:
    def __init__(
        self,
        repository: CalibrationCandidateRepository,
        crops: CropStore,
        policy: CapturePolicy,
        versions: CaptureVersions,
        attestation: AttestationPolicy = AttestationPolicy(),
    ) -> None:
        self._repository = repository
        self._crops = crops
        self._policy = policy
        self._versions = versions
        self._attestation = attestation
        self._held = AttestationCaptureBuffer(attestation)
        self._attested_counter = DailyBucketCounter()
        self._counter = DailyBucketCounter()
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue(maxsize=policy.queue_size)
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self._policy.enabled

    @property
    def attestation_enabled(self) -> bool:
        return self._attestation.enabled

    @property
    def held_for_attestation(self) -> int:
        return len(self._held)

    async def start(self) -> None:
        """Seed today's caps from stored captures and start the upload worker."""
        today = datetime.now(UTC).date()
        start_of_day = datetime.combine(today, time.min, tzinfo=UTC)
        if self._attestation.enabled:
            self._attested_counter.seed(
                today, await self._repository.bucket_counts(start_of_day, attested_only=True)
            )
        if not self._policy.enabled:
            return
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
        """Queue the faces of one identify call that the policy keeps; returns how many.

        Independently, with attestation enabled and a detection id in the context, every face
        that clears the attestation floors is held in memory for a later attestation.
        """
        if context is None:
            return 0
        if self._attestation.enabled and context.detection_id:
            self._hold_for_attestation(image, faces, identities, context)
        if not self._policy.enabled:
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

    def _hold_for_attestation(
        self,
        image: np.ndarray,
        faces: Sequence[DetectedFace],
        identities: Sequence[IdentifyResult],
        context: CaptureContext,
    ) -> None:
        assert context.detection_id is not None
        # The tracker attests a detection because it attributed one face in the person crop to
        # that person; it does not say which face. With two faces in the crop the held face
        # could be a neighbour's, so such a crop holds nothing and its attestation is a miss.
        if len(faces) > 1:
            attestation_faces_total.labels(result="skipped_multiple_faces").inc(len(faces))
            self._held.expire(context.captured_at)
            return
        for face, identity in zip(faces, identities, strict=True):
            reason = attestation_hold_rejection(face, self._attestation)
            if reason is not None:
                attestation_faces_total.labels(result=reason).inc()
                continue
            assert face.keypoints_5_px is not None  # checked by attestation_hold_rejection
            self._held.hold(
                context.detection_id,
                _Held(align_face(image, face.keypoints_5_px), face, identity, context),
            )
            attestation_faces_total.labels(result="held").inc()
        self._held.expire(context.captured_at)

    async def attest(
        self,
        *,
        association_id: str,
        identity_id: str,
        grade: AttestationGrade,
        detection_ids: Sequence[str],
        now: datetime,
    ) -> AttestationOutcome:
        """Store the held faces of these detections as attested candidates of *identity_id*.

        Goes through the same crop store and repository as operator captures. A detection whose
        face is no longer held (expired, never held, already attested) is a miss, not an error.
        """
        moved = missed = capped = 0
        for detection_id in dict.fromkeys(detection_ids):
            held = self._held.take(detection_id, now=now)
            if held is None:
                missed += 1
                continue
            assert held.face.quality is not None
            bucket = (quality_decile(held.face.quality.score), identity_id, held.context.camera_id)
            if not self._attested_counter.try_take(
                bucket, held.context.captured_at, self._attestation.per_bucket_per_day
            ):
                capped += 1
                continue
            candidate = replace(
                self._candidate(held.face, held.identity, held.context),
                label_state="attested",
                label_kind="member",
                label_identity_id=identity_id,
                label_source="sensor_attested",
                labeled_by=f"wearable:{association_id}",
                labeled_at=now,
                attestation_association_id=association_id,
                attestation_grade=grade,
            )
            try:
                await self._write(candidate, held.aligned, prefix="attested")
            except Exception:
                logger.exception("attestation_store_failed")
                attestation_faces_total.labels(result="failed").inc()
                continue
            moved += 1
        attestation_faces_total.labels(result="moved").inc(moved)
        attestation_faces_total.labels(result="missed").inc(missed)
        attestation_faces_total.labels(result="capped").inc(capped)
        return AttestationOutcome(moved=moved, missed=missed, capped=capped)

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
        face = pending.face
        # Guaranteed by should_capture before a face is queued.
        assert face.keypoints_5_px is not None
        aligned = align_face(pending.image, face.keypoints_5_px)
        await self._write(
            self._candidate(face, pending.identity, pending.context), aligned, prefix=""
        )

    async def _write(
        self, candidate: CalibrationCandidate, aligned: np.ndarray, *, prefix: str
    ) -> None:
        """Upload the aligned crop, then insert the row; never leaves an orphan crop."""
        day = f"{candidate.captured_at.astimezone(UTC):%Y-%m-%d}"
        folder = f"{CROP_PREFIX}/{prefix}" if prefix else CROP_PREFIX
        key = f"{folder}/{day}/{candidate.candidate_id}.png"
        ok, png = cv2.imencode(".png", aligned)
        if not ok:
            raise RuntimeError("could not encode aligned crop")
        await self._crops.put_png(key, png.tobytes())
        try:
            await self._repository.insert(replace(candidate, crop_object_key=key))
        except Exception:
            # Never leave an orphan crop for a row that does not exist.
            await self._crops.delete([key])
            raise

    def _candidate(
        self, face: DetectedFace, identity: IdentifyResult, context: CaptureContext
    ) -> CalibrationCandidate:
        """The candidate row for one face; ``_write`` fills in the crop key it uploads to."""
        assert face.quality is not None
        quality = face.quality
        return CalibrationCandidate(
            candidate_id=str(uuid.uuid4()),
            captured_at=context.captured_at,
            camera_id=context.camera_id,
            source="cts_identify",
            crop_object_key="",
            embedding=np.asarray(face.embedding, dtype=np.float32),
            arcface_model_version=self._versions.arcface_model_version,
            recognizer_version=self._versions.recognizer_version,
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

    async def purge_expired(self, now: datetime) -> int:
        """Delete unlabeled and rejected captures past retention, with their crops."""
        cutoff = now - timedelta(days=self._policy.unlabeled_retention_days)
        keys = await self._repository.purge(cutoff)
        await self._crops.delete(keys)
        return len(keys)

    async def delete_candidate(self, candidate_id: str) -> bool:
        return (await self.delete_candidate_returning_key(candidate_id)) is not None

    async def delete_candidate_returning_key(self, candidate_id: str) -> str | None:
        """Delete a candidate and its crop; returns the deleted crop key or None if missing."""
        key = await self._repository.delete(candidate_id)
        if key is None:
            return None
        await self._crops.delete([key])
        return key

    async def delete_for_person(self, person_id: str) -> int:
        """Delete candidates (and crops) associated with a removed member."""
        keys = await self._repository.delete_for_person(person_id)
        if keys:
            await self._crops.delete(keys)
        return len(keys)


class MinioCropStore:
    """``CropStore`` over the service's MinIO client (boto3 calls run in a worker thread)."""

    def __init__(self, client: MinioClient) -> None:
        self._client = client

    async def put_png(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self._client.upload_bytes, data, key, "image/png")

    async def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            await asyncio.to_thread(self._client.delete_object, key)

    async def get_bytes(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._client.download_bytes, key)

    async def delete_prefix(self, prefix: str) -> list[str]:
        return await asyncio.to_thread(self._client.delete_prefix, prefix)
