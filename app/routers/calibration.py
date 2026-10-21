"""Calibration-candidate review: list, label, reject, delete, and progress summary.

cognitive-companion proxies these for the operator's review screen and presigns the crop keys,
as it does for visitor clusters. Labels set ``label_source = operator_reviewed``.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from app.calibration.episodes import EpisodeInput, assign_episodes, episode_label
from app.models.calibration_review import (
    CalibrationSummary,
    CandidateListResponse,
    CandidateView,
    IdentityProgress,
    LabelRequest,
)
from app.services.calibration_candidates import CalibrationCandidate, quality_decile

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"])

# The fitter's per-identity holdout minimum and the holdout's share of episodes, used only for the
# progress hint in the summary.
_HOLDOUT_GENUINE_MINIMUM = 30
_HOLDOUT_FRACTION = 0.2


def _view(c: CalibrationCandidate) -> CandidateView:
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
    return CandidateListResponse(
        candidates=[_view(c) for c in page.candidates], next_cursor=page.next_cursor
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
    return _view(updated)


@router.post("/candidates/{candidate_id}/reject", response_model=CandidateView)
async def reject_candidate(request: Request, candidate_id: str) -> CandidateView:
    """Mark a candidate unusable; it is purged with unlabeled captures."""
    updated = await request.app.state.calibration_candidates.reject(candidate_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="candidate not found")
    return _view(updated)


@router.delete("/candidates/{candidate_id}", status_code=204)
async def delete_candidate(request: Request, candidate_id: str) -> None:
    """Delete a candidate and its crop (labeled candidates persist until deleted here)."""
    if not await request.app.state.calibration_capture.delete_candidate(candidate_id):
        raise HTTPException(status_code=404, detail="candidate not found")


@router.get("/summary", response_model=CalibrationSummary)
async def calibration_summary(request: Request) -> CalibrationSummary:
    """Label progress: counts per state, kind, quality decile and camera; episodes per identity."""
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
    )
