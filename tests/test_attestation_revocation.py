"""Wearable-attested face revocation: repository parity, routes, manifest, promotion, purge, migration."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pgvector.asyncpg import register_vector

from app.calibration.attestation_audit import MIN_AUDITED, AuditSummary
from app.calibration.manifest import GalleryEmbedding, ManifestMeta, household_records
from app.db.migrate import rollback, run_migrations, status
from app.routers import calibration
from app.services.calibration_candidates import (
    CalibrationCandidate,
    CalibrationCandidateRepository,
    InMemoryCalibrationCandidateRepository,
    PostgresCalibrationCandidateRepository,
)
from app.services.calibration_capture import (
    AttestationPolicy,
    CalibrationCaptureService,
    CapturePolicy,
)
from app.services.enrollment_repository import InMemoryEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore, PromotionIneligibleError
from app.services.match_policy import MatchThresholds
from tests.calibration_fakes import VERSIONS, InMemoryCropStore
from tests.db import safe_test_dsn

TEST_DSN = safe_test_dsn()
_T0 = datetime(2026, 6, 15, 10, 0, 0, tzinfo=UTC)


def _attested_candidate(
    *,
    index: int,
    association_id: str = "seg-1",
    audit_verdict: str | None = None,
    captured_at: datetime | None = None,
) -> CalibrationCandidate:
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    at = captured_at or (_T0 + timedelta(minutes=index))
    return CalibrationCandidate(
        candidate_id=f"11111111-1111-1111-1111-{index:012d}",
        captured_at=at,
        camera_id="cam-1",
        source="cts_identify",
        crop_object_key=f"calibration/attested/{association_id}-{index}.png",
        embedding=emb,
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        quality_version="heuristic-v1",
        quality_score=0.75,
        quality_components={"resolution": 0.8, "sharpness": 0.7, "detection": 0.9, "pitch": 1.0},
        capture_profile={"crop_pad_top_frac": 0.15},
        interocular_px=34.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        nearest_person_id="resident-alpha",
        nearest_similarity=0.65,
        label_state="attested",
        label_kind="member",
        label_identity_id="resident-alpha",
        label_source="sensor_attested",
        labeled_by=f"wearable:{association_id}",
        labeled_at=at,
        attestation_association_id=association_id,
        attestation_grade="trajectory",
        audit_verdict=audit_verdict,  # type: ignore[arg-type]
        audited_by="caregiver" if audit_verdict else None,
        audited_at=at if audit_verdict else None,
    )


def _build_capture_service(repo: CalibrationCandidateRepository) -> CalibrationCaptureService:
    return CalibrationCaptureService(
        repo,
        InMemoryCropStore(),
        CapturePolicy(enabled=False),
        VERSIONS,
        AttestationPolicy(enabled=True),
    )


async def _exercise_revocation_parity(repo: CalibrationCandidateRepository) -> None:
    c1 = _attested_candidate(index=1, association_id="seg-bad", audit_verdict=None)
    c2 = _attested_candidate(index=2, association_id="seg-bad", audit_verdict="correct")
    c3 = _attested_candidate(index=3, association_id="seg-good", audit_verdict="correct")
    for c in (c1, c2, c3):
        await repo.insert(c)

    # Before revocation all three appear in list_attested.
    assert [c.candidate_id for c in await repo.list_attested()] == [
        c1.candidate_id,
        c2.candidate_id,
        c3.candidate_id,
    ]

    revoked_at = _T0 + timedelta(hours=1)
    newly, all_for_seg = await repo.revoke_by_association(
        "seg-bad", revoked_by="admin", revoked_at=revoked_at
    )
    assert newly == 2
    assert [c.candidate_id for c in all_for_seg] == [c1.candidate_id, c2.candidate_id]
    assert all(c.revoked_by == "admin" and c.revoked_at is not None for c in all_for_seg)

    # Idempotent repeat returns newly == 0 but still returns all rows for the association.
    newly_again, all_again = await repo.revoke_by_association(
        "seg-bad", revoked_by="other", revoked_at=revoked_at + timedelta(minutes=5)
    )
    assert newly_again == 0
    assert len(all_again) == 2
    assert all(c.revoked_by == "admin" for c in all_again)

    # Revoked rows are excluded from default list_attested, included when include_revoked=True.
    assert [c.candidate_id for c in await repo.list_attested()] == [c3.candidate_id]
    assert [
        c.candidate_id
        for c in await repo.list_attested(association_id="seg-bad", include_revoked=True)
    ] == [c1.candidate_id, c2.candidate_id]

    # Auditing a revoked row is refused (returns None).
    assert (
        await repo.audit(
            c1.candidate_id, verdict="correct", audited_by="caregiver", audited_at=revoked_at
        )
        is None
    )

    # Purge removes revoked rows even if they were previously audited correct, while keeping
    # unrevoked audited-correct rows.
    purged_keys = await repo.purge(_T0 + timedelta(days=1))
    assert purged_keys == sorted([c1.crop_object_key, c2.crop_object_key])
    assert await repo.get(c3.candidate_id) is not None


async def test_revocation_inmemory_parity() -> None:
    await _exercise_revocation_parity(InMemoryCalibrationCandidateRepository())


async def test_manifest_and_promotion_exclude_revoked_attestations() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    c1 = _attested_candidate(index=10, association_id="seg-bad", audit_verdict="correct")
    c2 = _attested_candidate(index=11, association_id="seg-good", audit_verdict="correct")
    await repo.insert(c1)
    await repo.insert(c2)
    await repo.revoke_by_association(
        "seg-bad", revoked_by="admin", revoked_at=_T0 + timedelta(minutes=30)
    )

    revoked_c1 = await repo.get(c1.candidate_id)
    assert revoked_c1 is not None

    # Manifest builder excludes revoked rows even if caller passes them directly.
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    other = np.zeros(512, dtype=np.float32)
    other[1] = 1.0
    gallery = {
        "resident-alpha": [GalleryEmbedding(embedding=centroid)],
        "resident-beta": [GalleryEmbedding(embedding=other)],
    }
    audit = AuditSummary(
        attested=MIN_AUDITED + 2,
        audited=MIN_AUDITED,
        correct=MIN_AUDITED,
        wrong=0,
        not_a_face=0,
        precision_lower_bound=0.995,
    )
    meta = ManifestMeta()
    records = household_records(
        [revoked_c1, c2], gallery, episode_gap_s=300.0, meta=meta, attestation_audit=audit
    )
    assert meta.sensor_attested_included == 1
    assert len(records) == 2  # only c2 scored against the 2 enrolled identities

    # Promotion refuses a revoked capture.
    enroll_repo = InMemoryEnrollmentRepository()
    await enroll_repo.upsert_member("resident-alpha", "Resident Alpha")
    await enroll_repo.insert_embeddings("resident-alpha", [centroid])
    store = EnrollmentStore(
        enroll_repo,
        None,  # type: ignore[arg-type]
        MatchThresholds(recognized_min_similarity=0.4, candidate_min_similarity=0.25),
    )
    with pytest.raises(PromotionIneligibleError, match="attestation_revoked"):
        await store.promote_candidate(
            revoked_c1, approved_by="admin", active_quality_version="heuristic-v1"
        )


def test_revoke_and_counts_routes_report_already_promoted() -> None:
    repo = InMemoryCalibrationCandidateRepository()
    c_promoted = _attested_candidate(index=20, association_id="seg-swap", audit_verdict="correct")
    c_unpromoted = _attested_candidate(index=21, association_id="seg-swap", audit_verdict=None)
    c_other = _attested_candidate(index=22, association_id="seg-other", audit_verdict=None)

    enroll_repo = InMemoryEnrollmentRepository()
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    store = EnrollmentStore(
        enroll_repo,
        None,  # type: ignore[arg-type]
        MatchThresholds(recognized_min_similarity=0.4, candidate_min_similarity=0.25),
    )

    async def _setup() -> None:
        await enroll_repo.upsert_member("resident-alpha", "Resident Alpha")
        await enroll_repo.insert_embeddings("resident-alpha", [centroid])
        for c in (c_promoted, c_unpromoted, c_other):
            await repo.insert(c)
        # Operator deliberately promotes one exemplar before revocation.
        await store.promote_candidate(
            c_promoted, approved_by="caregiver", active_quality_version="heuristic-v1"
        )

    asyncio.run(_setup())

    app = FastAPI()
    app.include_router(calibration.router)
    app.state.calibration_candidates = repo
    app.state.calibration_capture = _build_capture_service(repo)
    app.state.enrollment_store = store
    client = TestClient(app)

    before_counts = client.get(
        "/api/v1/calibration/attestations/counts", params={"association_id": "seg-swap"}
    )
    assert before_counts.status_code == 200
    assert before_counts.json() == {
        "association_id": "seg-swap",
        "total": 2,
        "active": 2,
        "revoked": 0,
        "promoted": 1,
    }

    resp = client.post(
        "/api/v1/calibration/attestations/revoke",
        json={"association_id": "seg-swap", "revoked_by": "admin"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["association_id"] == "seg-swap"
    assert body["revoked"] == 2
    assert body["total_revoked"] == 2
    assert body["already_promoted"] == [
        {
            "candidate_id": c_promoted.candidate_id,
            "person_id": "resident-alpha",
            "pose_bin": "frontal",
        }
    ]

    # Promoted embedding stays untouched in enrollment; operator reviews it manually.
    promoted_ids = asyncio.run(enroll_repo.promoted_candidate_ids())
    assert c_promoted.candidate_id in promoted_ids

    # Revoked rows disappear from audit sample and exemplar grid; seg-other remains.
    sample = client.get("/api/v1/calibration/attestations/audit-sample").json()["candidates"]
    assert [c["candidate_id"] for c in sample] == [c_other.candidate_id]

    # Repeat revoke is idempotent and still surfaces already_promoted.
    repeat = client.post(
        "/api/v1/calibration/attestations/revoke",
        json={"association_id": "seg-swap", "revoked_by": "admin"},
    )
    assert repeat.status_code == 200
    assert repeat.json()["revoked"] == 0
    assert repeat.json()["total_revoked"] == 2
    assert len(repeat.json()["already_promoted"]) == 1

    after_counts = client.get(
        "/api/v1/calibration/attestations/counts", params={"association_id": "seg-swap"}
    ).json()
    assert after_counts == {
        "association_id": "seg-swap",
        "total": 2,
        "active": 0,
        "revoked": 2,
        "promoted": 1,
    }


@pytest_asyncio.fixture
async def db_pool() -> AsyncIterator[asyncpg.Pool]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=2)
    await run_migrations(pool)
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM calibration_candidates")
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM calibration_candidates")
    await run_migrations(pool)
    await pool.close()


async def test_revocation_postgres_parity(db_pool: asyncpg.Pool) -> None:
    await _exercise_revocation_parity(PostgresCalibrationCandidateRepository(db_pool))


async def test_migration_0007_refuses_while_revoked_and_round_trips(db_pool: asyncpg.Pool) -> None:
    repo = PostgresCalibrationCandidateRepository(db_pool)
    c = _attested_candidate(index=99, association_id="seg-mig")
    await repo.insert(c)
    await repo.revoke_by_association("seg-mig", revoked_by="admin", revoked_at=_T0)

    # Non-attested row with revoked_at violates the completeness constraint.
    with pytest.raises(asyncpg.CheckViolationError):
        async with db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO calibration_candidates (
                    candidate_id, captured_at, camera_id, source, crop_object_key, embedding,
                    arcface_model_version, preprocessing_version, quality_version, quality_score,
                    quality_components, capture_profile, yaw_deg, pitch_deg, nearest_similarity,
                    label_state, revoked_at, revoked_by)
                VALUES ($1::uuid, now(), 'cam-1', 'cts_identify', 'k', $2, 'buffalo_l', 'v1',
                        'heuristic-v1', 0.5, '{}'::jsonb, '{}'::jsonb, 0, 0, 0.4, 'unlabeled',
                        now(), 'admin')""",
                str(uuid.uuid4()),
                np.zeros(512, dtype=np.float32),
            )

    # Down migration refuses while revoked rows exist.
    with pytest.raises(asyncpg.RaiseError, match="revoked rows"):
        await rollback(db_pool, steps=1)

    # After clearing revoked rows, down and up round-trip cleanly.
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM calibration_candidates")
    await rollback(db_pool, steps=1)
    async with db_pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'calibration_candidates' AND column_name = 'revoked_at'"
            )
            == 0
        )
    await run_migrations(db_pool)
    assert "0007_attestation_revocation" in [s.version for s in await status(db_pool) if s.applied]
