"""Enrollment service: orchestrates the face engine, gallery repository, and matching policy.

It holds no SQL and reads no configuration. Persistence is an ``EnrollmentRepository``; turning
nearest identities into a recognition state is ``classify_match``; pose-binned prototype
construction is ``build_prototypes``; this module only sequences them and carries detection-level
fields onto results.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

import numpy as np

from app.models.enrollment import EnrollResult, MemberInfo
from app.observability.metrics import gallery_representation_shadow_total
from app.services.calibration_candidates import CalibrationCandidate
from app.services.enrollment_repository import EnrollmentRepository
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.match_policy import MatchDecision, MatchThresholds, classify_match
from app.services.prototype_builder import (
    MemberEmbedding,
    PoseBin,
    Prototype,
    PrototypePolicy,
    pose_bin,
    promotion_ineligibility,
)

logger = logging.getLogger(__name__)

GalleryRepresentationMode = Literal["centroid", "shadow", "prototypes"]
ACTIVE_REPRESENTATION_CENTROID = "centroid"
ACTIVE_REPRESENTATION_PROTOTYPES = "prototypes-v1"


class PromotionIneligibleError(ValueError):
    """Candidate does not meet promotion requirements (maps to HTTP 422)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class PromotionConflictError(ValueError):
    """Candidate was already promoted (maps to HTTP 409)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class PromotionResult:
    person_id: str
    pose_bin: PoseBin
    embedding_id: int
    gallery_revision: int
    prototypes: list[Prototype]


class FaceEngineProtocol(Protocol):
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]: ...


def shadow_comparison_outcome(centroid: MatchDecision, prototype: MatchDecision) -> str:
    """Classify the difference between a centroid decision and a prototype decision."""
    if (
        centroid.recognition_state == prototype.recognition_state
        and centroid.person_id == prototype.person_id
    ):
        return "same_state_same_id"
    if (
        centroid.person_id != prototype.person_id
        and centroid.recognition_state == "recognized"
        and prototype.recognition_state == "recognized"
    ):
        return "identity_changed"
    if centroid.recognition_state != prototype.recognition_state:
        return "state_changed"
    return "identity_changed"


class EnrollmentStore:
    """Enrolls household members and identifies faces against centroids or pose prototypes."""

    def __init__(
        self,
        repository: EnrollmentRepository,
        face_engine: FaceEngineProtocol,
        thresholds: MatchThresholds,
        *,
        gallery_representation: GalleryRepresentationMode = "centroid",
        prototype_policy: PrototypePolicy = PrototypePolicy(),
        on_member_removed: Callable[[str], Awaitable[object]] | None = None,
    ) -> None:
        self._repository = repository
        self._engine = face_engine
        self._thresholds = thresholds
        self._gallery_representation: GalleryRepresentationMode = gallery_representation
        self._prototype_policy = prototype_policy
        self._on_member_removed = on_member_removed

    @property
    def gallery_representation_mode(self) -> GalleryRepresentationMode:
        return self._gallery_representation

    @property
    def active_representation(self) -> str:
        """The representation serving live answers (``shadow`` serves ``centroid``)."""
        if self._gallery_representation == "prototypes":
            return ACTIVE_REPRESENTATION_PROTOTYPES
        return ACTIVE_REPRESENTATION_CENTROID

    @property
    def prototype_policy(self) -> PrototypePolicy:
        return self._prototype_policy

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

        records: list[MemberEmbedding] = []
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
            records.append(
                MemberEmbedding(
                    embedding=best_face.embedding,
                    pose_bin=pose_bin(best_face.yaw_deg, best_face.pitch_deg),
                    source="enrollment_photo",
                    quality_score=best_face.quality.score if best_face.quality else None,
                    quality_version=best_face.quality.version if best_face.quality else None,
                )
            )

        if not records:
            return EnrollResult(
                person_id=person_id,
                name=name,
                embedding_count=existing_count,
                status="failed",
                failed_images=failed_indices,
            )

        status = await self._repository.upsert_member(person_id, name)
        await self._repository.insert_embeddings(person_id, records)
        if status == "updated":
            await self._repository.bump_gallery_revision(person_id)
        await self._repository.recompute_centroid(person_id)
        await self._repository.rebuild_prototypes(person_id, self._prototype_policy)

        total_count = await self._repository.embedding_count(person_id)
        logger.info(
            "Enrolled person_id=%s embeddings=%d status=%s",
            person_id,
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
        """Identify one face embedding using the configured gallery representation.

        Always queries top 2 identities so top-2 margin is reported. In ``shadow`` mode, returns
        the centroid result while comparing against prototypes and updating the shadow counter.
        """
        if self._gallery_representation == "prototypes":
            hits = await self._repository.nearest_identities(embedding, k=2)
            decision = classify_match(hits, self._thresholds)
            rep = ACTIVE_REPRESENTATION_PROTOTYPES
        else:
            centroid_hits = await self._repository.nearest_centroids(embedding, k=2)
            decision = classify_match(centroid_hits, self._thresholds)
            rep = ACTIVE_REPRESENTATION_CENTROID
            if self._gallery_representation == "shadow":
                proto_hits = await self._repository.nearest_identities(embedding, k=2)
                proto_decision = classify_match(proto_hits, self._thresholds)
                outcome = shadow_comparison_outcome(decision, proto_decision)
                gallery_representation_shadow_total.labels(outcome=outcome).inc()

        return IdentifyResult(
            person_id=decision.person_id,
            name=decision.name,
            bbox=[],
            best_candidate_id=decision.best_candidate_id,
            similarity=decision.similarity,
            recognition_state=decision.recognition_state,
            second_best_candidate_id=decision.second_best_candidate_id,
            second_best_similarity=decision.second_best_similarity,
            margin=decision.margin,
            gallery_representation=rep,
            gallery_revision=decision.gallery_revision,
        )

    async def identify_all(self, faces: list[DetectedFace]) -> list[IdentifyResult]:
        """Identify every face in a frame, one gallery query per face."""
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

    async def promote_candidate(
        self,
        candidate: CalibrationCandidate,
        *,
        approved_by: str,
        active_quality_version: str,
        person_id: str | None = None,
    ) -> PromotionResult:
        """Promote an operator-labeled calibration capture into a member's gallery and prototypes.

        Requires a labeled ``member`` candidate passing quality version, quality floor, and
        interocular distance floor. Increments the member's ``gallery_revision`` so deployed
        calibration artifacts fail closed for that member until refitted.
        """
        promoted_ids = await self._repository.promoted_candidate_ids()
        already_promoted = candidate.candidate_id in promoted_ids
        reason = promotion_ineligibility(
            label_state=candidate.label_state,
            label_kind=candidate.label_kind,
            label_identity_id=candidate.label_identity_id,
            quality_score=candidate.quality_score,
            quality_version=candidate.quality_version,
            interocular_px=candidate.interocular_px,
            active_quality_version=active_quality_version,
            policy=self._prototype_policy,
            already_promoted=already_promoted,
            audit_verdict=candidate.audit_verdict,
            revoked=candidate.revoked_at is not None,
        )
        if reason == "already_promoted":
            raise PromotionConflictError(reason, "candidate has already been promoted")
        if reason is not None:
            raise PromotionIneligibleError(reason, f"candidate cannot be promoted: {reason}")

        assert candidate.label_identity_id is not None
        target_id = candidate.label_identity_id
        if person_id is not None and person_id != target_id:
            raise PromotionIneligibleError(
                "person_id_mismatch",
                "target person_id does not match the candidate's label_identity_id",
            )
        if not await self._repository.member_exists(target_id):
            raise PromotionIneligibleError(
                "member_not_enrolled",
                "labeled member is not currently enrolled",
            )

        target_bin = pose_bin(candidate.yaw_deg, candidate.pitch_deg)
        embedding_id = await self._repository.add_member_embedding(
            target_id,
            candidate.embedding,
            pose_bin=target_bin,
            source="deployment_capture",
            quality_score=candidate.quality_score,
            quality_version=candidate.quality_version,
            source_candidate_id=candidate.candidate_id,
            approved_by=approved_by,
            approved_at=datetime.now(UTC),
            crop_object_key=candidate.crop_object_key,
        )
        await self._repository.recompute_centroid(target_id)
        prototypes = await self._repository.rebuild_prototypes(target_id, self._prototype_policy)
        member = await self._repository.get_member(target_id)
        revision = member.gallery_revision if member is not None else 1
        logger.info(
            "Promoted candidate_id=%s person_id=%s pose_bin=%s gallery_revision=%d approved_by=%s",
            candidate.candidate_id,
            target_id,
            target_bin,
            revision,
            approved_by,
        )
        return PromotionResult(
            person_id=target_id,
            pose_bin=target_bin,
            embedding_id=embedding_id,
            gallery_revision=revision,
            prototypes=prototypes,
        )

    async def list_prototypes(self, person_id: str) -> list[Prototype]:
        return await self._repository.list_prototypes(person_id)

    async def rebuild_prototypes(self, person_id: str) -> list[Prototype]:
        return await self._repository.rebuild_prototypes(person_id, self._prototype_policy)

    async def delete_prototype(self, person_id: str, prototype_id: str) -> bool:
        return await self._repository.delete_prototype(
            person_id, prototype_id, self._prototype_policy
        )

    async def promoted_candidate_ids(self) -> set[str]:
        return await self._repository.promoted_candidate_ids()

    async def clear_crop_reference(self, crop_object_key: str) -> None:
        await self._repository.clear_crop_reference(crop_object_key)

    async def list_members(self) -> list[MemberInfo]:
        """All enrolled members with embedding counts and prototype counts."""
        return await self._repository.list_members()

    async def get_member(self, person_id: str) -> MemberInfo | None:
        return await self._repository.get_member(person_id)

    async def remove_member(self, person_id: str) -> bool:
        """Remove a member with all embeddings, prototypes, centroids, and calibration crops."""
        deleted = await self._repository.remove_member(person_id)
        if deleted:
            if self._on_member_removed is not None:
                await self._on_member_removed(person_id)
            logger.info("Removed member person_id=%s", person_id)
        return deleted

    async def member_count(self) -> int:
        return await self._repository.member_count()
