"""EnrollmentStore orchestration over the in-memory repository: no database needed."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_models import DetectedFace
from tests.gallery import DEFAULT_THRESHOLDS

_IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


def _unit(seed: int) -> np.ndarray:
    vector = np.random.RandomState(seed).randn(512)
    return (vector / np.linalg.norm(vector)).astype(np.float32)


class _QueueEngine:
    """One face per image from a queue; an empty queue means the image has no face."""

    def __init__(self) -> None:
        self.queue: list[list[DetectedFace]] = []

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return self.queue.pop(0) if self.queue else []


def _face(embedding: np.ndarray, width_px: float = 50.0) -> DetectedFace:
    return DetectedFace(bbox=[0.0, 0.0, width_px, width_px], embedding=embedding, det_score=0.9)


@pytest.fixture
def engine() -> _QueueEngine:
    return _QueueEngine()


@pytest.fixture
def store(engine: _QueueEngine) -> EnrollmentStore:
    return EnrollmentStore(InMemoryEnrollmentRepository(), engine, DEFAULT_THRESHOLDS)


async def test_enroll_then_identify_recognizes_the_member(
    store: EnrollmentStore, engine: _QueueEngine
) -> None:
    alpha = _unit(1)
    engine.queue = [[_face(alpha)]]
    result = await store.enroll("alpha", "Alpha", [_IMAGE])

    assert (result.status, result.embedding_count, result.failed_images) == ("enrolled", 1, [])
    identified = await store.identify(alpha)
    assert (identified.recognition_state, identified.person_id) == ("recognized", "alpha")
    assert identified.similarity == pytest.approx(1.0, abs=1e-6)


async def test_reenroll_updates_and_reports_failed_images(
    store: EnrollmentStore, engine: _QueueEngine
) -> None:
    engine.queue = [[_face(_unit(1))]]
    await store.enroll("alpha", "Alpha", [_IMAGE])

    engine.queue = [[_face(_unit(2))], []]
    result = await store.enroll("alpha", "Alpha Renamed", [_IMAGE, _IMAGE])

    assert (result.status, result.embedding_count, result.failed_images) == ("updated", 2, [1])
    [member] = await store.list_members()
    assert (member.name, member.embedding_count) == ("Alpha Renamed", 2)


async def test_enroll_without_any_face_writes_nothing(
    store: EnrollmentStore, engine: _QueueEngine
) -> None:
    result = await store.enroll("ghost", "Ghost", [_IMAGE, _IMAGE])

    assert (result.status, result.embedding_count, result.failed_images) == ("failed", 0, [0, 1])
    assert await store.member_count() == 0


async def test_largest_face_in_an_image_is_enrolled(
    store: EnrollmentStore, engine: _QueueEngine
) -> None:
    small, large = _unit(3), _unit(4)
    engine.queue = [[_face(small, width_px=20.0), _face(large, width_px=80.0)]]
    await store.enroll("alpha", "Alpha", [_IMAGE])

    assert (await store.identify(large)).recognition_state == "recognized"
    assert (await store.identify(small)).recognition_state != "recognized"


async def test_remove_member_forgets_the_identity(
    store: EnrollmentStore, engine: _QueueEngine
) -> None:
    alpha = _unit(1)
    engine.queue = [[_face(alpha)]]
    await store.enroll("alpha", "Alpha", [_IMAGE])

    assert await store.remove_member("alpha") is True
    assert await store.remove_member("alpha") is False
    identified = await store.identify(alpha)
    assert (identified.recognition_state, identified.best_candidate_id) == ("unrecognized", None)
