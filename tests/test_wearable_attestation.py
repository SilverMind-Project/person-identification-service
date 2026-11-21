"""Wearable-attested face exemplars: the in-memory hold, attestation, audit, and the manifest gate.

Attested faces are machine labels: a face is written only when tracking-orchestrator attests its
detection, it lands in state ``attested`` (never in the operator queue or enrollment), an
operator audits a uniform random sample, and the manifest builder refuses attested labels until
the audit shows at least 200 audited with a precision lower bound of 0.99 or better.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.calibration.attestation_audit import (
    MIN_AUDITED,
    AuditSummary,
    audit_sample,
    summarize,
    wilson_lower_bound,
)
from app.calibration.manifest import (
    GalleryEmbedding,
    ManifestError,
    ManifestMeta,
    household_records,
)
from app.routers import calibration
from app.services.calibration_candidates import (
    CalibrationCandidate,
    InMemoryCalibrationCandidateRepository,
)
from app.services.calibration_capture import (
    AttestationPolicy,
    CalibrationCaptureService,
    CaptureContext,
    CapturePolicy,
)
from tests.calibration_fakes import VERSIONS, InMemoryCropStore
from tests.gallery import one_member_store
from tests.test_calibration_capture_policy import _IMAGE, _face, _identity

_T = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)


def _context(detection_id: str | None, at: datetime = _T, camera: str = "cam-1") -> CaptureContext:
    return CaptureContext(
        camera_id=camera,
        captured_at=at,
        crop_pad_top_frac=0.15,
        crop_pad_side_frac=0.1,
        crop_jpeg_quality=95,
        detection_id=detection_id,
    )


def _service(
    repo: InMemoryCalibrationCandidateRepository,
    crops: InMemoryCropStore | None = None,
    *,
    capture: bool = False,
    **attestation: Any,
) -> CalibrationCaptureService:
    return CalibrationCaptureService(
        repo,
        crops or InMemoryCropStore(),
        CapturePolicy(enabled=capture),
        VERSIONS,
        replace(AttestationPolicy(enabled=True), **attestation),
    )


async def _attest(service: CalibrationCaptureService, *ids: str, at: datetime = _T) -> Any:
    return await service.attest(
        association_id="seg-1",
        identity_id="resident-alpha",
        grade="trajectory",
        detection_ids=list(ids),
        now=at + timedelta(seconds=5),
    )


async def test_held_faces_become_attested_member_labels_and_nothing_else_is_written() -> None:
    repo, crops = InMemoryCalibrationCandidateRepository(), InMemoryCropStore()
    service = _service(repo, crops)
    # Capture is off: holding writes nothing.
    assert service.offer(_IMAGE, [_face()], [_identity()], _context("det-1")) == 0
    service.offer(_IMAGE, [_face()], [_identity(best="someone-else")], _context("det-2"))
    assert service.held_for_attestation == 2
    assert await repo.count_by_state() == {
        "unlabeled": 0,
        "labeled": 0,
        "rejected": 0,
        "attested": 0,
    }

    outcome = await _attest(service, "det-1", "det-1", "det-unknown")
    assert (outcome.moved, outcome.missed, outcome.capped) == (1, 1, 0)
    (row,) = await repo.list_attested()
    assert row.label_state == "attested"
    assert (row.label_kind, row.label_identity_id) == ("member", "resident-alpha")
    assert row.label_source == "sensor_attested"
    assert row.labeled_by == "wearable:seg-1"
    assert (row.attestation_association_id, row.attestation_grade) == ("seg-1", "trajectory")
    assert row.crop_object_key in crops.objects
    assert row.crop_object_key.startswith("calibration/attested/")
    # Attested rows never appear in the operator's labeling queue or the labeled dataset.
    assert (await repo.list_page(state=None, limit=10, cursor=None)).candidates == []
    assert await repo.list_labeled() == []
    # A face is attested once: the second attestation of det-1 is a miss.
    assert (await _attest(service, "det-1")).missed == 1


def test_hold_floors_and_switches() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    service = _service(repo, min_quality=0.5, min_interocular_px=31.0)
    service.offer(_IMAGE, [_face(quality_score=0.45)], [_identity()], _context("low-quality"))
    service.offer(_IMAGE, [_face(quality_score=0.6)], [_identity()], _context("small-face"))
    service.offer(_IMAGE, [_face(scored=False)], [_identity()], _context("unscored"))
    assert service.held_for_attestation == 0  # 30 px between the eyes < 31 px floor

    no_id = _service(repo)
    no_id.offer(_IMAGE, [_face()], [_identity()], _context(None))
    assert no_id.held_for_attestation == 0

    off = CalibrationCaptureService(
        repo, InMemoryCropStore(), CapturePolicy(), VERSIONS, AttestationPolicy(enabled=False)
    )
    off.offer(_IMAGE, [_face()], [_identity()], _context("det-1"))
    assert off.held_for_attestation == 0


async def test_a_crop_with_two_faces_holds_nothing() -> None:
    """A neighbour's face in the same person crop must never be labeled as the wearer."""
    repo = InMemoryCalibrationCandidateRepository()
    service = _service(repo)
    service.offer(
        _IMAGE,
        [_face(), _face(quality_score=0.9)],
        [_identity(), _identity(best="someone-else")],
        _context("det-1"),
    )
    assert service.held_for_attestation == 0
    assert (await _attest(service, "det-1")).missed == 1
    assert await repo.list_attested() == []


async def test_hold_is_bounded_by_count_and_capture_time() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    service = _service(repo, max_items=2, max_age_s=60.0)
    for index in range(3):
        service.offer(_IMAGE, [_face()], [_identity()], _context(f"det-{index}"))
    assert service.held_for_attestation == 2
    # The oldest was evicted first.
    assert (await _attest(service, "det-0")).missed == 1
    # A held face older than max_age_s can no longer be attested.
    late = await service.attest(
        association_id="seg-1",
        identity_id="resident-alpha",
        grade="room",
        detection_ids=["det-1"],
        now=_T + timedelta(seconds=61),
    )
    assert (late.moved, late.missed) == (0, 1)
    # Newer calls expire stale faces on arrival.
    service.offer(_IMAGE, [_face()], [_identity()], _context("det-new", at=_T + timedelta(hours=1)))
    assert service.held_for_attestation == 1


async def test_daily_cap_per_bucket() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    service = _service(repo, per_bucket_per_day=1)
    for index in range(2):
        service.offer(_IMAGE, [_face()], [_identity()], _context(f"det-{index}"))
    outcome = await _attest(service, "det-0", "det-1")
    assert (outcome.moved, outcome.capped) == (1, 1)


async def test_a_failed_store_leaves_no_orphan_crop() -> None:
    class _FailingRepo(InMemoryCalibrationCandidateRepository):
        async def insert(self, candidate: CalibrationCandidate) -> None:
            raise RuntimeError("database down")

    crops = InMemoryCropStore()
    service = _service(_FailingRepo(), crops)
    service.offer(_IMAGE, [_face()], [_identity()], _context("det-1"))
    outcome = await _attest(service, "det-1")
    assert (outcome.moved, outcome.missed) == (0, 0)
    assert crops.objects == {}


def test_wilson_lower_bound_matches_known_values() -> None:
    assert wilson_lower_bound(0, 0) is None
    assert wilson_lower_bound(50, 100) == pytest.approx(0.40383, abs=1e-4)
    assert wilson_lower_bound(200, 200) == pytest.approx(0.98115, abs=1e-4)
    assert wilson_lower_bound(400, 400) == pytest.approx(0.99050, abs=1e-4)
    with pytest.raises(ValueError, match="within"):
        wilson_lower_bound(3, 2)


def _attested(index: int, verdict: str | None = None) -> CalibrationCandidate:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    return CalibrationCandidate(
        candidate_id=f"00000000-0000-0000-0000-{index:012d}",
        captured_at=_T + timedelta(minutes=index),
        camera_id="cam-1",
        source="cts_identify",
        crop_object_key=f"calibration/attested/{index}.png",
        embedding=embedding,
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        quality_version="heuristic-v1",
        quality_score=0.6,
        quality_components={"resolution": 0.6, "sharpness": 0.6, "detection": 0.9, "pitch": 1.0},
        capture_profile={"crop_pad_top_frac": 0.15},
        interocular_px=30.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        nearest_person_id="resident-alpha",
        nearest_similarity=0.5,
        label_state="attested",
        label_kind="member",
        label_identity_id="resident-alpha",
        label_source="sensor_attested",
        labeled_by="wearable:seg-1",
        labeled_at=_T,
        attestation_association_id="seg-1",
        attestation_grade="trajectory",
        audit_verdict=verdict,  # type: ignore[arg-type]
        audited_by="op" if verdict else None,
        audited_at=_T if verdict else None,
    )


def test_summary_and_bar() -> None:
    below = summarize([_attested(i, "correct") for i in range(199)])
    assert (below.audited, below.meets_bar) == (199, False)
    perfect = summarize([_attested(i, "correct") for i in range(400)])
    assert perfect.meets_bar is True
    one_wrong = summarize([_attested(i, "correct") for i in range(399)] + [_attested(399, "wrong")])
    assert one_wrong.meets_bar is False  # lower bound 0.986 < 0.99
    mixed = summarize([_attested(0, "not_a_face"), _attested(1), _attested(2, "wrong")])
    assert (mixed.attested, mixed.audited, mixed.not_a_face, mixed.wrong) == (3, 2, 1, 1)


def test_audit_sample_is_uniform_and_seeded() -> None:
    rows = [_attested(i) for i in range(20)]
    assert audit_sample(rows, 5, seed=7) == audit_sample(rows, 5, seed=7)
    assert len(audit_sample(rows, 50, seed=1)) == 20
    picks: Counter[str] = Counter()
    for seed in range(2000):
        picks.update(c.candidate_id for c in audit_sample(rows, 5, seed=seed))
    # Each row is drawn with probability 5/20; 2000 draws give 500 each in expectation.
    assert min(picks.values()) > 420
    assert max(picks.values()) < 580


def _gallery() -> dict[str, list[GalleryEmbedding]]:
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    other = np.zeros(512, dtype=np.float32)
    other[1] = 1.0
    return {
        "resident-alpha": [GalleryEmbedding(embedding=centroid)],
        "resident-beta": [GalleryEmbedding(embedding=other)],
    }


def test_manifest_refuses_attested_labels_below_the_audit_bar() -> None:
    rows = [_attested(i) for i in range(3)]
    with pytest.raises(ManifestError, match="audit"):
        household_records(rows, _gallery(), episode_gap_s=300.0, meta=ManifestMeta())
    weak = summarize([_attested(i, "correct") for i in range(150)])
    with pytest.raises(ManifestError, match="audit"):
        household_records(
            rows, _gallery(), episode_gap_s=300.0, meta=ManifestMeta(), attestation_audit=weak
        )


def test_manifest_admits_attested_labels_above_the_bar_except_audited_wrong() -> None:
    audit = AuditSummary(
        attested=MIN_AUDITED + 2,
        audited=MIN_AUDITED,
        correct=MIN_AUDITED,
        wrong=0,
        not_a_face=0,
        precision_lower_bound=0.995,
    )
    rows = [_attested(0), _attested(1, "correct"), _attested(2, "wrong")]
    meta = ManifestMeta()
    records = household_records(
        rows, _gallery(), episode_gap_s=300.0, meta=meta, attestation_audit=audit
    )
    assert meta.sensor_attested_included == 2
    assert meta.skipped_audited_wrong == 1
    assert meta.attestation_audit is not None
    assert meta.attestation_audit["meets_bar"] is True
    assert {r.label_source for r in records} == {"sensor_attested"}


@pytest.fixture
def client() -> TestClient:
    repo = InMemoryCalibrationCandidateRepository()
    for index, verdict in enumerate([None, None, "correct"]):
        asyncio.run(repo.insert(_attested(index, verdict)))
    app = FastAPI()
    app.include_router(calibration.router)
    app.state.calibration_candidates = repo
    app.state.calibration_capture = _service(repo)
    app.state.enrollment_store = one_member_store(
        None,  # type: ignore[arg-type]
        person_id="resident-alpha",
        name="Resident Alpha",
        probe=np.eye(512, dtype=np.float32)[0],
        similarity=0.5,
    )
    return TestClient(app)


def test_attestation_route_rejects_an_unknown_identity(client: TestClient) -> None:
    body = {
        "association_id": "seg-1",
        "identity_id": "nobody",
        "attestation_grade": "room",
        "detection_ids": ["det-1"],
        "attested_by": "wearable",
    }
    response = client.post("/api/v1/calibration/attestations", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unknown_identity"
    ok = client.post(
        "/api/v1/calibration/attestations", json=body | {"identity_id": "resident-alpha"}
    )
    assert ok.json() == {"moved": 0, "missed": 1, "capped": 0}
    extra = client.post("/api/v1/calibration/attestations", json=body | {"note": "x"})
    assert extra.status_code == 422


def test_audit_routes(client: TestClient) -> None:
    sample = client.get(
        "/api/v1/calibration/attestations/audit-sample", params={"n": 5, "seed": 3}
    ).json()["candidates"]
    assert len(sample) == 2  # only the unaudited ones
    assert all(c["label_state"] == "attested" and c["audit_verdict"] is None for c in sample)

    target = sample[0]["candidate_id"]
    audited = client.post(
        f"/api/v1/calibration/attestations/{target}/audit",
        json={"verdict": "wrong", "audited_by": "caregiver"},
    )
    assert audited.status_code == 200
    assert (audited.json()["audit_verdict"], audited.json()["audited_by"]) == ("wrong", "caregiver")
    assert audited.json()["promotion_ineligible_reason"] == "audited_wrong"
    missing = client.post(
        "/api/v1/calibration/attestations/00000000-0000-0000-0000-000000000999/audit",
        json={"verdict": "correct", "audited_by": "caregiver"},
    )
    assert missing.status_code == 404

    summary = client.get("/api/v1/calibration/attestations/audit-summary").json()
    assert (summary["attested"], summary["audited"], summary["correct"], summary["wrong"]) == (
        3,
        2,
        1,
        1,
    )
    assert summary["min_audited"] == MIN_AUDITED
    assert summary["meets_bar"] is False

    grid = client.get(
        "/api/v1/calibration/attestations", params={"identity_id": "resident-alpha"}
    ).json()["candidates"]
    assert len(grid) == 3
    correct = next(c for c in grid if c["audit_verdict"] == "correct")
    assert correct["promotion_eligible"] is True
