"""Calibration-candidate review: list, label, promote, reject, delete, and progress summary.

cognitive-companion proxies these for the operator's review screen and presigns the crop keys,
as it does for visitor clusters. Labels set ``label_source = operator_reviewed`` and never change
the enrollment gallery; only explicit operator promotion via ``POST /candidates/{id}/promote``
adds an embedding to the gallery and rebuilds that member's prototypes.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from app.calibration.attestation_audit import (
    MIN_AUDITED,
    MIN_PRECISION_LOWER_BOUND,
    audit_sample,
    summarize,
)
from app.calibration.episodes import EpisodeInput, assign_episodes, episode_label
from app.models.calibration_review import (
    AttestationRequest,
    AttestationResponse,
    AttestedCandidateList,
    AuditRequest,
    AuditSummaryResponse,
    CalibrationSummary,
    CandidateListResponse,
    CandidateView,
    IdentityProgress,
    LabelBatchRequest,
    LabelBatchResponse,
    LabelBatchResult,
    LabelRequest,
    PromoteCandidateRequest,
    PromoteCandidateResponse,
)
from app.services.calibration_candidates import CalibrationCandidate, quality_decile
from app.services.enrollment_store import PromotionConflictError, PromotionIneligibleError
from app.services.prototype_builder import PrototypePolicy, pose_bin, promotion_ineligibility

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"])

# The fitter's per-identity holdout minimum and the holdout's share of episodes, used only for the
# progress hint in the summary.
_HOLDOUT_GENUINE_MINIMUM = 30
_HOLDOUT_FRACTION = 0.2


def _active_quality_version(request: Request) -> str:
    return str(getattr(request.app.state, "face_quality_version", "heuristic-v1"))


def _prototype_policy(request: Request) -> PrototypePolicy:
    store = getattr(request.app.state, "enrollment_store", None)
    if store is not None and hasattr(store, "prototype_policy"):
        return store.prototype_policy
    return PrototypePolicy()


def _view(
    c: CalibrationCandidate,
    *,
    promoted_ids: set[str],
    active_quality_version: str,
    policy: PrototypePolicy,
) -> CandidateView:
    promoted = c.candidate_id in promoted_ids
    reason = promotion_ineligibility(
        label_state=c.label_state,
        label_kind=c.label_kind,
        label_identity_id=c.label_identity_id,
        quality_score=c.quality_score,
        quality_version=c.quality_version,
        interocular_px=c.interocular_px,
        active_quality_version=active_quality_version,
        policy=policy,
        already_promoted=promoted,
        audit_verdict=c.audit_verdict,
    )
    return CandidateView(
        candidate_id=c.candidate_id,
        captured_at=c.captured_at,
        camera_id=c.camera_id,
        crop_object_key=c.crop_object_key,
        quality_score=c.quality_score,
        quality_version=c.quality_version,
        interocular_px=c.interocular_px,
        yaw_deg=c.yaw_deg,
        pitch_deg=c.pitch_deg,
        nearest_person_id=c.nearest_person_id,
        nearest_similarity=c.nearest_similarity,
        label_state=c.label_state,
        label_kind=c.label_kind,
        label_identity_id=c.label_identity_id,
        labeled_by=c.labeled_by,
        labeled_at=c.labeled_at,
        pose_bin=pose_bin(c.yaw_deg, c.pitch_deg),
        promoted=promoted,
        promotion_eligible=reason is None,
        promotion_ineligible_reason=reason,
        attestation_association_id=c.attestation_association_id,
        attestation_grade=c.attestation_grade,
        audit_verdict=c.audit_verdict,
        audited_by=c.audited_by,
        audited_at=c.audited_at,
    )


@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(
    request: Request,
    state: Literal["unlabeled", "labeled", "rejected"] | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = None,
) -> CandidateListResponse:
    """Candidates newest first, one page at a time."""
    repo = request.app.state.calibration_candidates
    try:
        page = await repo.list_page(state=state, limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    promoted_ids = await request.app.state.enrollment_store.promoted_candidate_ids()
    quality_version = _active_quality_version(request)
    policy = _prototype_policy(request)
    return CandidateListResponse(
        candidates=[
            _view(
                c,
                promoted_ids=promoted_ids,
                active_quality_version=quality_version,
                policy=policy,
            )
            for c in page.candidates
        ],
        next_cursor=page.next_cursor,
    )


@router.post("/candidates/{candidate_id}/label", response_model=CandidateView)
async def label_candidate(request: Request, candidate_id: str, body: LabelRequest) -> CandidateView:
    """Label a candidate as a household member, a person who is not enrolled, or not a face."""
    if body.label_identity_id is not None:
        member = await request.app.state.enrollment_store.get_member(body.label_identity_id)
        if member is None:
            raise HTTPException(
                status_code=422, detail="label_identity_id is not an enrolled member"
            )
    updated = await request.app.state.calibration_candidates.label(
        candidate_id,
        kind=body.label_kind,
        identity_id=body.label_identity_id,
        source="operator_reviewed",
        labeled_by=body.labeled_by,
        labeled_at=datetime.now(UTC),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    promoted_ids = await request.app.state.enrollment_store.promoted_candidate_ids()
    return _view(
        updated,
        promoted_ids=promoted_ids,
        active_quality_version=_active_quality_version(request),
        policy=_prototype_policy(request),
    )


@router.post("/candidates/label-batch", response_model=LabelBatchResponse)
async def label_candidates(request: Request, body: LabelBatchRequest) -> LabelBatchResponse:
    """Label several candidates as someone not enrolled, or not a face; one result each.

    Each label is its own write, so one missing candidate never undoes the others.
    """
    repo = request.app.state.calibration_candidates
    labeled_at = datetime.now(UTC)
    results: list[LabelBatchResult] = []
    for candidate_id in body.candidate_ids:
        updated = await repo.label(
            candidate_id,
            kind=body.label_kind,
            identity_id=None,
            source="operator_reviewed",
            labeled_by=body.labeled_by,
            labeled_at=labeled_at,
        )
        results.append(
            LabelBatchResult(
                candidate_id=candidate_id,
                ok=updated is not None,
                error_code=None if updated is not None else "not_found",
            )
        )
    labeled = sum(1 for r in results if r.ok)
    return LabelBatchResponse(results=results, labeled=labeled, failed=len(results) - labeled)


@router.post("/candidates/{candidate_id}/promote", response_model=PromoteCandidateResponse)
async def promote_candidate(
    request: Request, candidate_id: str, body: PromoteCandidateRequest
) -> PromoteCandidateResponse:
    """Promote a labeled member capture into that member's gallery and rebuild prototypes.

    Side effect: adds the capture's embedding to ``embeddings`` with ``source=deployment_capture``,
    increments the member's ``gallery_revision`` (pausing calibrated recognition authority for that
    member until the next calibration fit), recomputes the centroid, and rebuilds prototypes.
    """
    candidate = await request.app.state.calibration_candidates.get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="candidate not found")

    try:
        result = await request.app.state.enrollment_store.promote_candidate(
            candidate,
            approved_by=body.approved_by,
            active_quality_version=_active_quality_version(request),
            person_id=body.person_id,
        )
    except PromotionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PromotionIneligibleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    counts_by_bin: dict[str, int] = {}
    for proto in result.prototypes:
        counts_by_bin[proto.pose_bin] = counts_by_bin.get(proto.pose_bin, 0) + 1
    return PromoteCandidateResponse(
        candidate_id=candidate_id,
        person_id=result.person_id,
        pose_bin=result.pose_bin,
        embedding_id=result.embedding_id,
        gallery_revision=result.gallery_revision,
        prototype_count=len(result.prototypes),
        counts_by_bin=dict(sorted(counts_by_bin.items())),
    )


@router.post("/candidates/{candidate_id}/reject", response_model=CandidateView)
async def reject_candidate(request: Request, candidate_id: str) -> CandidateView:
    """Mark a candidate unusable; it is purged with unlabeled captures."""
    updated = await request.app.state.calibration_candidates.reject(candidate_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    promoted_ids = await request.app.state.enrollment_store.promoted_candidate_ids()
    return _view(
        updated,
        promoted_ids=promoted_ids,
        active_quality_version=_active_quality_version(request),
        policy=_prototype_policy(request),
    )


@router.delete("/candidates/{candidate_id}", status_code=204)
async def delete_candidate(request: Request, candidate_id: str) -> None:
    """Delete a candidate and its crop (labeled candidates persist until deleted here).

    If the candidate was already promoted, its promoted embedding and source candidate id stay in
    the gallery; only the deleted crop reference is cleared.
    """
    deleted_key = await request.app.state.calibration_capture.delete_candidate_returning_key(
        candidate_id
    )
    if deleted_key is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    store = getattr(request.app.state, "enrollment_store", None)
    if store is not None and hasattr(store, "clear_crop_reference"):
        await store.clear_crop_reference(deleted_key)


@router.get("/summary", response_model=CalibrationSummary)
async def calibration_summary(request: Request) -> CalibrationSummary:
    """Label progress: counts per state, kind, quality decile, camera, and promotable pose bins."""
    repo = request.app.state.calibration_candidates
    labeled = await repo.list_labeled()
    usable = [c for c in labeled if c.label_kind in ("member", "not_enrolled")]
    episodes = assign_episodes(
        EpisodeInput(
            candidate_id=c.candidate_id,
            captured_at=c.captured_at,
            camera_id=c.camera_id,
            label=episode_label(c.label_kind, c.label_identity_id),
        )
        for c in usable
    )
    faces = Counter(c.label_identity_id for c in usable if c.label_kind == "member")
    identity_episodes: dict[str, set[str]] = {}
    for c in usable:
        if c.label_kind == "member" and c.label_identity_id is not None:
            identity_episodes.setdefault(c.label_identity_id, set()).add(episodes[c.candidate_id])

    quality_version = _active_quality_version(request)
    policy = _prototype_policy(request)
    promotable: dict[str, dict[str, int]] = {}
    for c in usable:
        if c.label_kind != "member" or c.label_identity_id is None:
            continue
        reason = promotion_ineligibility(
            label_state=c.label_state,
            label_kind=c.label_kind,
            label_identity_id=c.label_identity_id,
            quality_score=c.quality_score,
            quality_version=c.quality_version,
            interocular_px=c.interocular_px,
            active_quality_version=quality_version,
            policy=policy,
        )
        if reason is None:
            b = pose_bin(c.yaw_deg, c.pitch_deg)
            by_bin = promotable.setdefault(c.label_identity_id, {})
            by_bin[b] = by_bin.get(b, 0) + 1

    target = math.ceil(_HOLDOUT_GENUINE_MINIMUM / _HOLDOUT_FRACTION)
    return CalibrationSummary(
        by_state=await repo.count_by_state(),
        by_label_kind=dict(Counter(str(c.label_kind) for c in labeled)),
        by_quality_decile=dict(Counter(str(quality_decile(c.quality_score)) for c in labeled)),
        by_camera=dict(Counter(c.camera_id for c in labeled)),
        identities=[
            IdentityProgress(
                identity_id=identity,
                labeled_faces=faces[identity],
                episodes=len(identity_episodes[identity]),
                target_labeled_faces=target,
            )
            for identity in sorted(identity_episodes)
        ],
        not_enrolled_episodes=len(
            {episodes[c.candidate_id] for c in usable if c.label_kind == "not_enrolled"}
        ),
        capture_enabled=request.app.state.calibration_capture.enabled,
        promotable_by_identity_pose={
            pid: dict(sorted(bins.items())) for pid, bins in sorted(promotable.items())
        },
    )


async def _views(request: Request, candidates: list[CalibrationCandidate]) -> list[CandidateView]:
    promoted_ids = await request.app.state.enrollment_store.promoted_candidate_ids()
    quality_version = _active_quality_version(request)
    policy = _prototype_policy(request)
    return [
        _view(c, promoted_ids=promoted_ids, active_quality_version=quality_version, policy=policy)
        for c in candidates
    ]


@router.post("/attestations", response_model=AttestationResponse)
async def attest_faces(request: Request, body: AttestationRequest) -> AttestationResponse:
    """Keep the held faces of these detections as wearable-attested captures of a member.

    Side effect: each face still held in memory for a listed detection is stored (aligned crop in
    MinIO, row in ``calibration_candidates``) in state ``attested`` with ``label_source =
    sensor_attested``. Nothing reaches enrollment or recognition: an operator audits a sample and
    promotes exemplars explicitly. Detection ids without a held face are counted as missed.
    """
    if await request.app.state.enrollment_store.get_member(body.identity_id) is None:
        raise HTTPException(
            status_code=422,
            detail={"code": "unknown_identity", "message": "identity is not enrolled"},
        )
    capture = request.app.state.calibration_capture
    if not capture.attestation_enabled:
        return AttestationResponse(moved=0, missed=len(body.detection_ids), capped=0)
    outcome = await capture.attest(
        association_id=body.association_id,
        identity_id=body.identity_id,
        grade=body.attestation_grade,
        detection_ids=body.detection_ids,
        now=datetime.now(UTC),
    )
    return AttestationResponse(moved=outcome.moved, missed=outcome.missed, capped=outcome.capped)


@router.get("/attestations", response_model=AttestedCandidateList)
async def list_attested(
    request: Request,
    identity_id: str | None = None,
    audited: bool | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> AttestedCandidateList:
    """Attested captures, best face quality first (the exemplar grid an operator promotes from)."""
    rows = await request.app.state.calibration_candidates.list_attested(
        audited=audited, identity_id=identity_id
    )
    rows.sort(key=lambda c: (-c.quality_score, c.captured_at))
    return AttestedCandidateList(candidates=await _views(request, rows[:limit]))


@router.get("/attestations/audit-sample", response_model=AttestedCandidateList)
async def attestation_audit_sample(
    request: Request,
    n: int = Query(default=50, ge=1, le=200),
    seed: int | None = None,
) -> AttestedCandidateList:
    """A uniform random sample of unaudited attested captures, so the auditor cannot choose."""
    unaudited = await request.app.state.calibration_candidates.list_attested(audited=False)
    return AttestedCandidateList(
        candidates=await _views(request, audit_sample(unaudited, n, seed=seed))
    )


@router.post("/attestations/{candidate_id}/audit", response_model=CandidateView)
async def audit_attestation(
    request: Request, candidate_id: str, body: AuditRequest
) -> CandidateView:
    """Record whether an attested label is right. Changes no label, gallery, or calibration."""
    updated = await request.app.state.calibration_candidates.audit(
        candidate_id,
        verdict=body.verdict,
        audited_by=body.audited_by,
        audited_at=datetime.now(UTC),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="attested candidate not found")
    return (await _views(request, [updated]))[0]


@router.get("/attestations/audit-summary", response_model=AuditSummaryResponse)
async def attestation_audit_summary(request: Request) -> AuditSummaryResponse:
    """Audit counts and the precision lower bound the manifest builder's gate reads."""
    summary = summarize(await request.app.state.calibration_candidates.list_attested())
    return AuditSummaryResponse(
        attested=summary.attested,
        audited=summary.audited,
        correct=summary.correct,
        wrong=summary.wrong,
        not_a_face=summary.not_a_face,
        precision_lower_bound=summary.precision_lower_bound,
        min_audited=MIN_AUDITED,
        min_precision_lower_bound=MIN_PRECISION_LOWER_BOUND,
        meets_bar=summary.meets_bar,
    )
