"""Candidate promotion, prototype CRUD, leave-episode-out exclusion, and member-delete cleanup."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.manifest import GalleryEmbedding, ManifestMeta, household_records
from app.routers import calibration, enrollment
from app.services.calibration_candidates import (
    CalibrationCandidate,
    InMemoryCalibrationCandidateRepository,
)
from app.services.calibration_capture import CalibrationCaptureService, CapturePolicy
from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_models import DetectedFace
from app.services.match_policy import MatchThresholds
from app.services.prototype_builder import MemberEmbedding, PrototypePolicy
from app.tools.rebuild_prototypes import rebuild_all_prototypes
from tests.calibration_fakes import VERSIONS, InMemoryCropStore

_T = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


def _axis(index: int) -> np.ndarray:
    v = np.zeros(512, dtype=np.float32)
    v[index] = 1.0
    return v


class _EmptyEngine:
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return []


def _candidate(
    idx: int,
    *,
    label_state: str = "labeled",
    label_kind: str | None = "member",
    label_identity_id: str | None = "alice",
    nearest_person_id: str | None = "alice",
    quality_score: float = 0.72,
    quality_version: str = "heuristic-v1",
    interocular_px: float | None = 30.0,
    yaw_deg: float = 0.0,
    pitch_deg: float = 24.0,
    minutes: float = 0.0,
    embedding: np.ndarray | None = None,
) -> CalibrationCandidate:
    return CalibrationCandidate(
        candidate_id=str(uuid.UUID(int=idx + 100)),
        captured_at=_T + timedelta(minutes=minutes),
        camera_id="cam-1",
        source="cts_identify",
        crop_object_key=f"calibration/2026-11-19/{idx}.png",
        embedding=_axis(idx + 1) if embedding is None else embedding,
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        quality_version=quality_version,
        quality_score=quality_score,
        quality_components={"resolution": 0.7, "sharpness": 0.7, "detection": 0.9, "pitch": 0.6},
        capture_profile={"crop_jpeg_quality": 95},
        interocular_px=interocular_px,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        nearest_person_id=nearest_person_id,
        nearest_similarity=0.5,
        label_state=label_state,  # type: ignore[arg-type]
        label_kind=label_kind,  # type: ignore[arg-type]
        label_identity_id=label_identity_id,
        label_source="operator_reviewed" if label_state == "labeled" else None,
        labeled_by="op" if label_state == "labeled" else None,
        labeled_at=_T if label_state == "labeled" else None,
    )


async def _app() -> tuple[
    TestClient,
    InMemoryEnrollmentRepository,
    InMemoryCalibrationCandidateRepository,
    InMemoryCropStore,
    EnrollmentStore,
]:
    enroll_repo = InMemoryEnrollmentRepository()
    await enroll_repo.upsert_member("alice", "Alice")
    await enroll_repo.insert_embeddings(
        "alice", [MemberEmbedding(embedding=_axis(0), pose_bin="frontal")]
    )
    await enroll_repo.recompute_centroid("alice")
    await enroll_repo.rebuild_prototypes("alice")

    cand_repo = InMemoryCalibrationCandidateRepository()
    crops = InMemoryCropStore()
    capture = CalibrationCaptureService(cand_repo, crops, CapturePolicy(enabled=False), VERSIONS)
    store = EnrollmentStore(
        enroll_repo,
        _EmptyEngine(),
        MatchThresholds(recognized_min_similarity=0.4, candidate_min_similarity=0.25),
        gallery_representation="prototypes",
        prototype_policy=PrototypePolicy(
            min_promotion_quality=0.5, min_promotion_interocular_px=24.0
        ),
        on_member_removed=capture.delete_for_person,
    )

    app = FastAPI()
    app.include_router(calibration.router)
    app.include_router(enrollment.router)
    app.state.calibration_candidates = cand_repo
    app.state.calibration_capture = capture
    app.state.enrollment_store = store
    app.state.face_quality_version = "heuristic-v1"
    return TestClient(app), enroll_repo, cand_repo, crops, store


async def test_promote_validates_gates_records_approver_and_rebuilds_prototypes() -> None:
    client, enroll_repo, cand_repo, crops, _ = await _app()
    good = _candidate(1, pitch_deg=25.0)
    low_q = _candidate(2, quality_score=0.35)
    small_iod = _candidate(3, interocular_px=18.0)
    wrong_ver = _candidate(4, quality_version="heuristic-v2")
    not_member = _candidate(5, label_kind="not_enrolled", label_identity_id=None)
    unlabeled = _candidate(6, label_state="unlabeled", label_kind=None, label_identity_id=None)

    for c in (good, low_q, small_iod, wrong_ver, not_member, unlabeled):
        await cand_repo.insert(c)
        crops.objects[c.crop_object_key] = b"png"

    # Reject ineligible candidates with 422.
    for bad in (low_q, small_iod, wrong_ver, not_member, unlabeled):
        resp = client.post(
            f"/api/v1/calibration/candidates/{bad.candidate_id}/promote",
            json={"approved_by": "caregiver-admin"},
        )
        assert resp.status_code == 422

    # Promote the eligible candidate.
    ok = client.post(
        f"/api/v1/calibration/candidates/{good.candidate_id}/promote",
        json={"approved_by": "caregiver-admin"},
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["person_id"] == "alice"
    assert body["pose_bin"] == "down"
    assert body["gallery_revision"] == 2
    assert body["counts_by_bin"] == {"down": 1, "frontal": 1}

    # Promoting the same candidate again fails with 409.
    dup = client.post(
        f"/api/v1/calibration/candidates/{good.candidate_id}/promote",
        json={"approved_by": "caregiver-admin"},
    )
    assert dup.status_code == 409

    # List prototypes shows the promoted prototype with approver and crop key.
    protos = client.get("/api/v1/members/alice/prototypes").json()
    assert protos["gallery_revision"] == 2
    down_proto = next(p for p in protos["prototypes"] if p["pose_bin"] == "down")
    assert down_proto["source"] == "deployment_capture"
    assert down_proto["approved_by"] == "caregiver-admin"
    assert down_proto["crop_object_key"] == good.crop_object_key

    # Summary reports promotable labeled captures per identity and pose bin.
    summary = client.get("/api/v1/calibration/summary").json()
    assert summary["promotable_by_identity_pose"] == {"alice": {"down": 1}}

    # Deleting the promoted candidate removes its crop and clears crop_object_key, keeping the
    # promoted embedding and source_candidate_id intact.
    del_resp = client.delete(f"/api/v1/calibration/candidates/{good.candidate_id}")
    assert del_resp.status_code == 204
    assert good.crop_object_key not in crops.objects
    records = (await enroll_repo.member_embedding_records())["alice"]
    promoted_rec = next(r for r in records if r.source_candidate_id == good.candidate_id)
    assert promoted_rec.crop_object_key is None
    protos_after = client.get("/api/v1/members/alice/prototypes").json()
    down_after = next(p for p in protos_after["prototypes"] if p["pose_bin"] == "down")
    assert down_after["crop_object_key"] is None

    # Removing the promoted prototype deletes its embedding and bumps revision; removing the last
    # remaining prototype is refused with 409.
    rm_down = client.delete(f"/api/v1/members/alice/prototypes/{down_after['prototype_id']}")
    assert rm_down.status_code == 200
    remaining = client.get("/api/v1/members/alice/prototypes").json()
    assert remaining["gallery_revision"] == 3
    assert len(remaining["prototypes"]) == 1
    last_proto_id = remaining["prototypes"][0]["prototype_id"]
    rm_last = client.delete(f"/api/v1/members/alice/prototypes/{last_proto_id}")
    assert rm_last.status_code == 409


async def test_promoted_candidate_is_excluded_from_episode_mates_in_prototype_manifest() -> None:
    _, enroll_repo, cand_repo, _, store = await _app()
    # Two captures in the same walk (1 minute apart), sharing axis 1 (down pose).
    c1 = _candidate(1, pitch_deg=25.0, minutes=0.0, embedding=_axis(1))
    c2 = _candidate(2, pitch_deg=25.0, minutes=1.0, embedding=_axis(1))
    await cand_repo.insert(c1)
    await cand_repo.insert(c2)

    await store.promote_candidate(c1, approved_by="admin", active_quality_version="heuristic-v1")

    records_by_member = await enroll_repo.member_embedding_records()
    gallery = {
        pid: [
            GalleryEmbedding(
                embedding=e.embedding,
                source_candidate_id=e.source_candidate_id,
                pose_bin=e.pose_bin,
                source=e.source,
                quality_score=e.quality_score,
                quality_version=e.quality_version,
            )
            for e in items
        ]
        for pid, items in records_by_member.items()
    }
    meta = ManifestMeta()
    manifest_records = household_records(
        [c1, c2],
        gallery,
        episode_gap_s=300.0,
        meta=meta,
        gallery_representation="prototypes-v1",
    )
    assert meta.gallery_representation == "prototypes-v1"
    # c2 shares c1's episode, so c1's promoted axis-1 embedding is excluded; only alice's frontal
    # axis-0 photo remains, giving similarity 0.0 instead of 1.0!
    c2_genuine = next(
        r for r in manifest_records if r.probe_id == c2.candidate_id and r.identity_id == "alice"
    )
    assert c2_genuine.similarity == pytest.approx(0.0, abs=1e-6)


async def test_remove_member_purges_associated_calibration_candidates_and_crops() -> None:
    client, _, cand_repo, crops, _ = await _app()
    alice_labeled = _candidate(1, label_identity_id="alice", nearest_person_id="bob")
    alice_unlabeled = _candidate(
        2,
        label_state="unlabeled",
        label_kind=None,
        label_identity_id=None,
        nearest_person_id="alice",
    )
    # A labeled not_enrolled capture whose nearest guess happened to be alice must stay (not alice's face).
    visitor_labeled = _candidate(
        3, label_kind="not_enrolled", label_identity_id=None, nearest_person_id="alice"
    )
    for c in (alice_labeled, alice_unlabeled, visitor_labeled):
        await cand_repo.insert(c)
        crops.objects[c.crop_object_key] = b"png"

    resp = client.delete("/api/v1/members/alice")
    assert resp.status_code == 200
    assert await cand_repo.get(alice_labeled.candidate_id) is None
    assert await cand_repo.get(alice_unlabeled.candidate_id) is None
    assert await cand_repo.get(visitor_labeled.candidate_id) is not None
    assert alice_labeled.crop_object_key not in crops.objects
    assert alice_unlabeled.crop_object_key not in crops.objects
    assert visitor_labeled.crop_object_key in crops.objects


async def test_rebuild_all_prototypes_tool_backfills_null_pose_and_is_idempotent() -> None:
    repo = InMemoryEnrollmentRepository()
    await repo.upsert_member("alice", "Alice")
    await repo.insert_embeddings("alice", [_axis(0), _axis(1)])  # pose_bin is None
    await repo.recompute_centroid("alice")

    first_backfilled, first_counts = await rebuild_all_prototypes(repo)
    assert first_backfilled == 2
    assert first_counts == {"alice": 2}

    second_backfilled, second_counts = await rebuild_all_prototypes(repo)
    assert second_backfilled == 0
    assert second_counts == {"alice": 2}
