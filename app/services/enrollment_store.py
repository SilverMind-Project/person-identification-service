"""Enrollment service: orchestrates the face engine, the gallery repository, and matching policy.

It holds no SQL and reads no configuration. Persistence is an ``EnrollmentRepository``; turning
nearest centroids into a recognition state is ``classify_match``; this module only sequences them
and carries detection-level fields onto results.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from app.models.enrollment import EnrollResult, MemberInfo
from app.services.enrollment_repository import EnrollmentRepository
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.match_policy import MatchThresholds, classify_match

logger = logging.getLogger(__name__)


class FaceEngineProtocol(Protocol):
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]: ...


class EnrollmentStore:
    """Enrolls household members and identifies faces against their centroids."""

    def __init__(
        self,
        repository: EnrollmentRepository,
        face_engine: FaceEngineProtocol,
        thresholds: MatchThresholds,
    ) -> None:
        self._repository = repository
        self._engine = face_engine
        self._thresholds = thresholds

    async def enroll(self, person_id: str, name: str, images: list[np.ndarray]) -> EnrollResult:
        """Enroll a new person or add images to an existing member.

        The largest face in each image is used; images without a face are reported in
        ``failed_images``. When no image yields a face nothing is written and the status is
        ``failed``.
        """
        existing_count = (
            await self._repository.embedding_count(person_id)
            if await self._repository.member_exists(person_id)
            else 0
        )

        embeddings: list[np.ndarray] = []
        failed_indices: list[int] = []
        for idx, img in enumerate(images):
            faces = await self._engine.detect_faces(img)
            if not faces:
                failed_indices.append(idx)
                continue
            best_face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            embeddings.append(best_face.embedding)

        if not embeddings:
            return EnrollResult(
                person_id=person_id,
                name=name,
                embedding_count=existing_count,
                status="failed",
                failed_images=failed_indices,
            )

        status = await self._repository.upsert_member(person_id, name)
        await self._repository.insert_embeddings(person_id, embeddings)
        await self._repository.recompute_centroid(person_id)

        total_count = await self._repository.embedding_count(person_id)
        logger.info(
            "Enrolled person_id=%s name=%s embeddings=%d status=%s",
            person_id,
            name,
            total_count,
            status,
        )
        return EnrollResult(
            person_id=person_id,
            name=name,
            embedding_count=total_count,
            status=status,
            failed_images=failed_indices,
        )

    async def identify(self, embedding: np.ndarray) -> IdentifyResult:
        """Identify one face embedding against the nearest enrolled centroid.

        Always returns the nearest member as ``best_candidate_id`` (except for an empty gallery),
        whatever the recognition state; see ``app.services.match_policy``.
        """
        decision = classify_match(
            await self._repository.nearest_centroids(embedding, k=1), self._thresholds
        )
        return IdentifyResult(
            person_id=decision.person_id,
            name=decision.name,
            bbox=[],
            best_candidate_id=decision.best_candidate_id,
            similarity=decision.similarity,
            recognition_state=decision.recognition_state,
        )

    async def identify_all(self, faces: list[DetectedFace]) -> list[IdentifyResult]:
        """Identify every face in a frame, one nearest-centroid query per face."""
        results: list[IdentifyResult] = []
        for face in faces:
            result = await self.identify(face.embedding)
            # Carry forward detection-level fields from the detected face.
            result.bbox = face.bbox
            result.yaw_deg = face.yaw_deg
            result.pitch_deg = face.pitch_deg
            result.roll_deg = face.roll_deg
            result.det_score = face.det_score
            result.face_width_px = face.geometry.face_width_px
            result.face_height_px = face.geometry.face_height_px
            result.interocular_px = face.geometry.interocular_px
            result.quality = face.quality
            results.append(result)
        return results

    async def list_members(self) -> list[MemberInfo]:
        """All enrolled members with embedding counts."""
        return await self._repository.list_members()

    async def get_member(self, person_id: str) -> MemberInfo | None:
        return await self._repository.get_member(person_id)

    async def remove_member(self, person_id: str) -> bool:
        """Remove a member with all their embeddings and their centroid."""
        deleted = await self._repository.remove_member(person_id)
        if deleted:
            logger.info("Removed member person_id=%s", person_id)
        return deleted

    async def member_count(self) -> int:
        return await self._repository.member_count()
