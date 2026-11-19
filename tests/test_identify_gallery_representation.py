"""Gallery representation modes (centroid, shadow, prototypes) on /identify and /match-template."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.evaluator import CalibrationEvaluator
from app.observability.metrics import gallery_representation_shadow_total
from app.routers import identification, templates
from app.services.calibration_candidates import InMemoryCalibrationCandidateRepository
from app.services.calibration_capture import CalibrationCaptureService, CapturePolicy
from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality
from app.services.match_policy import MatchThresholds
from app.services.prototype_builder import MemberEmbedding
from app.services.template_aggregation import QualityWeightedMeanAggregator, encode_embedding_f16
from app.services.template_match import TemplateMatchService, TemplateMember
from tests.calibration_fakes import VERSIONS, InMemoryCropStore


def _axis(index: int, weight: float = 1.0) -> np.ndarray:
    vec = np.zeros(512, dtype=np.float32)
    vec[index] = weight
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    vec = alpha * a + (1.0 - alpha) * b
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class _StaticEngine:
    model_profile = "full"

    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = faces

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return list(self._faces)


class _DisabledVisitors:
    async def record_sighting(self, *args, **kwargs):
        return None


def _quality() -> FaceQuality:
    return FaceQuality(
        score=0.8, resolution=0.8, sharpness=0.8, detection=0.9, pitch=0.9, version="heuristic-v1"
    )


async def _seed_gallery(repo: InMemoryEnrollmentRepository) -> tuple[np.ndarray, np.ndarray]:
    """Alice has 5 frontal embeddings on axis 0 and 1 downward embedding on axis 1.
    Bob has 3 frontal embeddings on axis 2."""
    frontal_vec = _axis(0)
    down_vec = _axis(1)
    bob_vec = _axis(2)

    await repo.upsert_member("alice", "Alice")
    await repo.insert_embeddings(
        "alice",
        [
            *[MemberEmbedding(embedding=frontal_vec, pose_bin="frontal") for _ in range(5)],
            MemberEmbedding(embedding=down_vec, pose_bin="down"),
        ],
    )
    await repo.recompute_centroid("alice")
    await repo.rebuild_prototypes("alice")

    await repo.upsert_member("bob", "Bob")
    await repo.insert_embeddings(
        "bob",
        [MemberEmbedding(embedding=bob_vec, pose_bin="frontal") for _ in range(3)],
    )
    await repo.recompute_centroid("bob")
    await repo.rebuild_prototypes("bob")
    return down_vec, bob_vec


async def test_downward_probe_scores_higher_against_down_prototype_than_frontal_centroid() -> None:
    repo = InMemoryEnrollmentRepository()
    down_vec, _ = await _seed_gallery(repo)
    thresholds = MatchThresholds(recognized_min_similarity=0.40, candidate_min_similarity=0.25)

    centroid_store = EnrollmentStore(
        repo, _StaticEngine([]), thresholds, gallery_representation="centroid"
    )
    proto_store = EnrollmentStore(
        repo, _StaticEngine([]), thresholds, gallery_representation="prototypes"
    )

    cent = await centroid_store.identify(down_vec)
    proto = await proto_store.identify(down_vec)

    # Centroid is 5 parts axis 0 + 1 part axis 1 -> similarity ~ 1/sqrt(26) ~= 0.196 (unrecognized).
    assert cent.gallery_representation == "centroid"
    assert cent.recognition_state == "unrecognized"
    assert cent.similarity < 0.25

    # Prototype has a dedicated `down` prototype on axis 1 -> similarity = 1.0 (recognized).
    assert proto.gallery_representation == "prototypes-v1"
    assert proto.recognition_state == "recognized"
    assert proto.person_id == "alice"
    assert proto.similarity == pytest.approx(1.0, abs=1e-5)
    assert proto.second_best_candidate_id == "bob"
    assert proto.margin == pytest.approx(1.0, abs=1e-5)


async def test_shadow_mode_answers_with_centroid_and_counts_every_outcome() -> None:
    repo = InMemoryEnrollmentRepository()
    down_vec, bob_vec = await _seed_gallery(repo)
    thresholds = MatchThresholds(recognized_min_similarity=0.40, candidate_min_similarity=0.25)
    shadow_store = EnrollmentStore(
        repo, _StaticEngine([]), thresholds, gallery_representation="shadow"
    )

    def _count(outcome: str) -> float:
        return gallery_representation_shadow_total.labels(outcome=outcome)._value.get()

    before = {k: _count(k) for k in ("same_state_same_id", "state_changed", "identity_changed")}

    # 1. Frontal Alice probe: both recognize Alice -> same_state_same_id.
    res_same = await shadow_store.identify(_axis(0))
    assert res_same.person_id == "alice"
    assert res_same.gallery_representation == "centroid"
    assert _count("same_state_same_id") == before["same_state_same_id"] + 1

    # 2. Downward Alice probe: centroid is unrecognized, prototype recognizes Alice -> state_changed.
    res_state = await shadow_store.identify(down_vec)
    assert res_state.recognition_state == "unrecognized"  # shadow answers with centroid
    assert _count("state_changed") == before["state_changed"] + 1

    # 3. Blend favoring Bob on centroid (0.6 Bob + 0.8 Alice-down, normalized):
    #    Centroid: dot(bob)=0.6 (recognized bob), dot(alice)=0.8/sqrt(26)~0.157.
    #    Prototype: dot(alice-down)=0.8 (recognized alice) > dot(bob)=0.6 -> identity_changed!
    probe_conflict = _blend(down_vec, bob_vec, 0.57)
    res_id = await shadow_store.identify(probe_conflict)
    assert res_id.person_id == "bob"  # centroid winner
    assert _count("identity_changed") == before["identity_changed"] + 1


async def test_template_match_supports_prototypes_and_reports_margin() -> None:
    repo = InMemoryEnrollmentRepository()
    down_vec, _ = await _seed_gallery(repo)
    thresholds = MatchThresholds(recognized_min_similarity=0.40, candidate_min_similarity=0.25)
    agg = QualityWeightedMeanAggregator()

    centroid_svc = TemplateMatchService(repo, agg, thresholds, gallery_representation="centroid")
    proto_svc = TemplateMatchService(repo, agg, thresholds, gallery_representation="prototypes")
    members = [TemplateMember(embedding=down_vec, quality=_quality()) for _ in range(3)]

    cent = await centroid_svc.match(members)
    proto = await proto_svc.match(members)

    assert cent.gallery_representation == "centroid"
    assert cent.decision.recognition_state == "unrecognized"
    assert proto.gallery_representation == "prototypes-v1"
    assert proto.decision.recognition_state == "recognized"
    assert proto.decision.person_id == "alice"
    assert proto.second_best_candidate_id == "bob"
    assert proto.margin == pytest.approx(1.0, abs=1e-5)


async def test_identify_and_match_template_wire_margin_and_representation() -> None:
    repo = InMemoryEnrollmentRepository()
    down_vec, _ = await _seed_gallery(repo)
    thresholds = MatchThresholds(recognized_min_similarity=0.40, candidate_min_similarity=0.25)
    face = DetectedFace(bbox=[10.0, 10.0, 90.0, 90.0], embedding=down_vec, det_score=0.95)
    face.quality = _quality()

    app = FastAPI()
    app.include_router(identification.router)
    app.include_router(templates.router)
    app.state.face_engine = _StaticEngine([face])
    app.state.enrollment_store = EnrollmentStore(
        repo, app.state.face_engine, thresholds, gallery_representation="prototypes"
    )
    app.state.template_matcher = TemplateMatchService(
        repo, QualityWeightedMeanAggregator(), thresholds, gallery_representation="prototypes"
    )
    app.state.face_quality_version = "heuristic-v1"
    app.state.calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.template_calibration_evaluator = CalibrationEvaluator(None, "degraded_missing")
    app.state.visitor_store = _DisabledVisitors()
    app.state.calibration_capture = CalibrationCaptureService(
        InMemoryCalibrationCandidateRepository(),
        InMemoryCropStore(),
        CapturePolicy(enabled=False),
        VERSIONS,
    )

    _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), dtype=np.uint8))
    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    client = TestClient(app)

    id_resp = client.post("/api/v1/identify", json={"image": img_b64})
    assert id_resp.status_code == 200
    det = id_resp.json()["faces"][0]
    assert det["person_id"] == "alice"
    assert det["second_best_candidate_id"] == "bob"
    assert det["second_best_similarity"] == pytest.approx(0.0, abs=1e-5)
    assert det["margin"] == pytest.approx(1.0, abs=1e-5)
    assert det["gallery_representation"] == "prototypes-v1"

    emb_b64 = encode_embedding_f16(down_vec)
    q_dict = {
        "score": 0.8,
        "resolution": 0.8,
        "sharpness": 0.8,
        "detection": 0.9,
        "pitch": 0.9,
        "version": "heuristic-v1",
    }
    tpl_resp = client.post(
        "/api/v1/match-template",
        json={
            "members": [
                {"embedding_f16_b64": emb_b64, "quality": q_dict},
                {"embedding_f16_b64": emb_b64, "quality": q_dict},
            ],
            "embedding_model_version": "buffalo_l",
        },
    )
    assert tpl_resp.status_code == 200
    tpl = tpl_resp.json()
    assert tpl["person_id"] == "alice"
    assert tpl["second_best_candidate_id"] == "bob"
    assert tpl["margin"] == pytest.approx(1.0, abs=1e-5)
    assert tpl["gallery_representation"] == "prototypes-v1"
