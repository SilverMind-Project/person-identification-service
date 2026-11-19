"""Template matching: several faces of one tracked person identified together."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app import config
from app.models.templates import (
    MatchTemplateRequest,
    MatchTemplateResponse,
    TemplateQualityModel,
)
from app.observability.metrics import template_matches_total
from app.services.face_quality import FaceQuality
from app.services.template_aggregation import decode_embedding_f16
from app.services.template_match import TemplateMember

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["identification"])


def _unprocessable(code: str, message: str) -> HTTPException:
    return HTTPException(status_code=422, detail={"code": code, "message": message})


@router.post("/match-template", response_model=MatchTemplateResponse)
async def match_template(request: Request, body: MatchTemplateRequest) -> MatchTemplateResponse:
    """Match 2 to 16 faces of one person, aggregated into one template, against the gallery.

    The caller sends embeddings it obtained from ``/identify`` with ``return_embedding`` and the
    quality objects from the same responses. The template is calibrated only by a ``template``
    profile artifact, never by the single-frame one. No side effects: nothing is offered to visitor
    clustering or calibration capture, because no image is received.
    """
    matcher = request.app.state.template_matcher
    evaluator = request.app.state.template_calibration_evaluator
    quality_version: str = request.app.state.face_quality_version
    active_arcface_version = str(config.get("calibration.arcface_model_version", ""))

    if body.embedding_model_version != active_arcface_version:
        raise _unprocessable(
            "template.embedding_model_version_mismatch",
            "embeddings come from a different ArcFace model than this service runs",
        )
    if any(m.quality.version != quality_version for m in body.members):
        raise _unprocessable(
            "template.quality_version_mismatch",
            "every member needs a quality score from this service's active scorer",
        )
    try:
        members = [
            TemplateMember(
                embedding=decode_embedding_f16(m.embedding_f16_b64),
                quality=FaceQuality(
                    score=m.quality.score,
                    resolution=m.quality.resolution,
                    sharpness=m.quality.sharpness,
                    detection=m.quality.detection,
                    pitch=m.quality.pitch,
                    version=m.quality.version,
                ),
            )
            for m in body.members
        ]
    except ValueError as exc:
        raise _unprocessable("template.embedding_invalid", str(exc)) from exc
    try:
        match = await matcher.match(members)
    except ValueError as exc:
        raise _unprocessable("template.aggregation_invalid", str(exc)) from exc

    decision = match.decision
    calibrated_confidence: float | None = None
    if decision.recognition_state == "recognized":
        features: dict[str, float] = {
            "similarity": decision.similarity,
            "quality_score": match.quality.score,
            "resolution": match.quality.resolution,
            "sharpness": match.quality.sharpness,
            "detection": match.quality.detection,
            "pitch": match.quality.pitch,
        }
        if match.margin is not None:
            features["margin"] = match.margin
        calibrated_confidence = evaluator.predict(
            features,
            decision.best_candidate_id,
            gallery_revision=decision.gallery_revision,
        )
    template_matches_total.labels(recognition_state=decision.recognition_state).inc()
    logger.info(
        "POST /match-template members=%d state=%s calibrated=%s",
        match.member_count,
        decision.recognition_state,
        calibrated_confidence is not None,
    )
    return MatchTemplateResponse(
        person_id=decision.person_id,
        name=decision.name,
        recognition_state=decision.recognition_state,
        best_candidate_id=decision.best_candidate_id,
        raw_similarity=decision.similarity,
        second_best_candidate_id=match.second_best_candidate_id,
        second_best_similarity=match.second_best_similarity,
        margin=match.margin,
        gallery_representation=match.gallery_representation,
        member_count=match.member_count,
        aggregator_version=match.aggregator_version,
        template_quality=TemplateQualityModel(
            score=match.quality.score,
            resolution=match.quality.resolution,
            sharpness=match.quality.sharpness,
            detection=match.quality.detection,
            pitch=match.quality.pitch,
        ),
        quality_version=quality_version,
        calibrated_confidence=calibrated_confidence,
        calibration_profile=evaluator.calibration_profile(),
        calibration_status=evaluator.health(),
        calibration_artifact_version=evaluator.artifact_version(),
    )
