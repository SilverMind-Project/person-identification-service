"""Template matching: several faces of one person matched as one aggregated embedding.

Sequences the aggregator, the gallery repository, and the matching policy, like
``EnrollmentStore.identify`` does for one face; it holds no SQL and reads no configuration.
Template thresholds are their own ``MatchThresholds`` (``recognition.template_*``) because an
aggregate of several faces scores higher against its own gallery than any single member does.
It has no side effects: it receives embeddings, never images, so nothing is offered to visitor
clustering or calibration capture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from app.observability.metrics import gallery_representation_shadow_total
from app.services.enrollment_repository import EnrollmentRepository
from app.services.enrollment_store import (
    ACTIVE_REPRESENTATION_CENTROID,
    ACTIVE_REPRESENTATION_PROTOTYPES,
    GalleryRepresentationMode,
    shadow_comparison_outcome,
)
from app.services.face_quality import FaceQuality
from app.services.match_policy import MatchDecision, MatchThresholds, classify_match
from app.services.template_aggregation import (
    TemplateAggregator,
    TemplateQuality,
    template_quality,
)


@dataclass(frozen=True)
class TemplateMember:
    """One face of the template: its ArcFace embedding (never logged) and its quality."""

    embedding: npt.NDArray[np.float32] = field(repr=False, compare=False)
    quality: FaceQuality


@dataclass(frozen=True)
class TemplateMatch:
    decision: MatchDecision
    # Raw cosine similarity of the template to the second-nearest identity; None for a gallery of
    # one member. Not a probability.
    second_best_similarity: float | None
    member_count: int
    aggregator_version: str
    quality: TemplateQuality
    second_best_candidate_id: str | None = None
    margin: float | None = None
    gallery_representation: str = ACTIVE_REPRESENTATION_CENTROID


class TemplateMatchService:
    """Aggregates member faces into a template and classifies it against the enrolled gallery."""

    def __init__(
        self,
        repository: EnrollmentRepository,
        aggregator: TemplateAggregator,
        thresholds: MatchThresholds,
        *,
        gallery_representation: GalleryRepresentationMode = "centroid",
    ) -> None:
        self._repository = repository
        self._aggregator = aggregator
        self._thresholds = thresholds
        self._gallery_representation: GalleryRepresentationMode = gallery_representation

    @property
    def aggregator_version(self) -> str:
        return self._aggregator.version

    @property
    def active_representation(self) -> str:
        if self._gallery_representation == "prototypes":
            return ACTIVE_REPRESENTATION_PROTOTYPES
        return ACTIVE_REPRESENTATION_CENTROID

    async def match(self, members: Sequence[TemplateMember]) -> TemplateMatch:
        """Match the members' template; ``ValueError`` when they cannot form one.

        Members are weighted by their quality score, so a sharp frontal face outweighs a blurred
        glimpse; the template's quality features come from ``template_quality``.
        """
        if not members:
            raise ValueError("a template needs at least one member")
        template = self._aggregator.aggregate(
            np.stack([np.asarray(m.embedding, dtype=np.float32) for m in members]),
            np.asarray([m.quality.score for m in members], dtype=np.float32),
        )
        if self._gallery_representation == "prototypes":
            hits = await self._repository.nearest_identities(template, k=2)
            decision = classify_match(hits, self._thresholds)
            rep = ACTIVE_REPRESENTATION_PROTOTYPES
        else:
            hits = await self._repository.nearest_centroids(template, k=2)
            decision = classify_match(hits, self._thresholds)
            rep = ACTIVE_REPRESENTATION_CENTROID
            if self._gallery_representation == "shadow":
                proto_hits = await self._repository.nearest_identities(template, k=2)
                proto_decision = classify_match(proto_hits, self._thresholds)
                outcome = shadow_comparison_outcome(decision, proto_decision)
                gallery_representation_shadow_total.labels(outcome=outcome).inc()

        return TemplateMatch(
            decision=decision,
            second_best_similarity=decision.second_best_similarity,
            second_best_candidate_id=decision.second_best_candidate_id,
            margin=decision.margin,
            gallery_representation=rep,
            member_count=len(members),
            aggregator_version=self._aggregator.version,
            quality=template_quality([m.quality for m in members]),
        )
