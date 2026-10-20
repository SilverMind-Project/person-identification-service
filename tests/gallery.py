"""A real ``EnrollmentStore`` over an in-memory gallery, for router and contract tests."""

from __future__ import annotations

import asyncio
import math

import numpy as np

from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore, FaceEngineProtocol
from app.services.match_policy import MatchThresholds

# The service defaults (recognition.threshold, recognition.unknown_threshold).
DEFAULT_THRESHOLDS = MatchThresholds(recognized_min_similarity=0.4, candidate_min_similarity=0.25)


def one_member_store(
    engine: FaceEngineProtocol,
    *,
    person_id: str,
    name: str,
    probe: np.ndarray,
    similarity: float,
) -> EnrollmentStore:
    """A store whose only member's centroid has exactly ``similarity`` cosine to unit ``probe``."""
    probe = np.asarray(probe, dtype=np.float64)
    orthogonal = np.zeros_like(probe)
    orthogonal[int(np.argmin(np.abs(probe)))] = 1.0
    orthogonal -= float(np.dot(orthogonal, probe)) * probe
    orthogonal /= np.linalg.norm(orthogonal)
    centroid = similarity * probe + math.sqrt(1.0 - similarity**2) * orthogonal

    repository = InMemoryEnrollmentRepository()

    async def _seed() -> None:
        await repository.upsert_member(person_id, name)
        await repository.insert_embeddings(person_id, [centroid.astype(np.float32)])
        await repository.recompute_centroid(person_id)

    asyncio.run(_seed())
    return EnrollmentStore(repository, engine, DEFAULT_THRESHOLDS)
