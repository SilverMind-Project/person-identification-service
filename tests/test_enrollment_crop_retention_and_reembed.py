"""Tests for enrollment crop retention, member-delete cleanup, and parallel-gallery reembed."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime

import asyncpg
import cv2
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import rollback, run_migrations, status
from app.services.calibration_candidates import CalibrationCandidate
from app.services.enrollment_repository import (
    InMemoryEnrollmentRepository,
    PostgresEnrollmentRepository,
)
from app.services.enrollment_store import EnrollmentStore, enrollment_crop_key
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality
from app.services.recognizer_profile import (
    ADAFACE_WEBFACE4M_LICENCE,
    BUFFALO_L_PROFILE,
    RecognizerProfile,
)
from app.tools.reembed import reembed_gallery, validate_reembed_profiles
from tests.db import safe_test_dsn
from tests.gallery import DEFAULT_THRESHOLDS

TEST_DSN = safe_test_dsn()

_ADAFACE_PROFILE = RecognizerProfile(
    name="adaface_ir50_webface4m",
    recognizer_version="adaface_ir50_webface4m",
    detector_model="face-detector-scrfd",
    recognition_model="face-recognition-adaface-ir50",
    landmark_2d_model="face-landmark-2d106",
    landmark_3d_model="face-landmark-3d68",
    genderage_model="face-attribute-genderage",
    color_space="bgr",
    recognition_input_name="input",
    recognition_output_name="embedding",
    licence=ADAFACE_WEBFACE4M_LICENCE,
)


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _png_bytes(fill: int = 120) -> bytes:
    img = np.full((112, 112, 3), fill, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class _InMemoryCropStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put_png(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def get_bytes(self, key: str) -> bytes | None:
        return self.objects.get(key)

    async def delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.objects.pop(key, None)

    async def delete_prefix(self, prefix: str) -> list[str]:
        removed = sorted(k for k in self.objects if k.startswith(prefix))
        for k in removed:
            del self.objects[k]
        return removed


class _QueueFaceEngine:
    def __init__(self, faces: list[DetectedFace]) -> None:
        self._faces = list(faces)

    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]:
        return [self._faces.pop(0)] if self._faces else []


class _DeterministicEmbedder:
    """Deterministic orthogonal-ish 512-dim embedding derived from the mean crop intensity."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed_aligned_crop(self, aligned_bgr: np.ndarray) -> np.ndarray:
        self.calls += 1
        seed = round(float(aligned_bgr.mean())) + 1000
        return _unit(seed)


async def test_enroll_and_promote_retain_crops_and_remove_member_deletes_them() -> None:
    repo = InMemoryEnrollmentRepository()
    crops = _InMemoryCropStore()
    removed_callbacks: list[str] = []

    async def _on_removed(pid: str) -> None:
        removed_callbacks.append(pid)

    crop_a = np.full((112, 112, 3), 80, dtype=np.uint8)
    crop_b = np.full((112, 112, 3), 160, dtype=np.uint8)
    face_a = DetectedFace(
        bbox=[10.0, 10.0, 90.0, 90.0],
        embedding=_unit(1),
        det_score=0.95,
        aligned_crop_bgr=crop_a,
    )
    face_b = DetectedFace(
        bbox=[10.0, 10.0, 90.0, 90.0],
        embedding=_unit(2),
        det_score=0.92,
        aligned_crop_bgr=crop_b,
    )
    store = EnrollmentStore(
        repo,
        _QueueFaceEngine([face_a, face_b]),
        DEFAULT_THRESHOLDS,
        on_member_removed=_on_removed,
        crop_store=crops,
    )
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    res = await store.enroll("alice", "Alice", [img, img])
    assert res.status == "enrolled"
    assert res.embedding_count == 2

    records = (await repo.member_embedding_records())["alice"]
    assert len(records) == 2
    for rec in records:
        assert rec.embedding_id is not None
        expected_key = enrollment_crop_key("alice", rec.embedding_id)
        assert rec.crop_object_key == expected_key
        assert expected_key in crops.objects
        decoded = cv2.imdecode(
            np.frombuffer(crops.objects[expected_key], np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == (112, 112, 3)

    # Promote a labeled calibration capture: its aligned crop is copied under enrollment/alice/.
    cal_key = "calibration/2026-11-21/cand-1.png"
    await crops.put_png(cal_key, _png_bytes(fill=200))
    cand = CalibrationCandidate(
        candidate_id=str(uuid.uuid4()),
        captured_at=datetime(2026, 11, 21, 12, 0, tzinfo=UTC),
        camera_id="cam-hall",
        source="cts_identify",
        crop_object_key=cal_key,
        embedding=_unit(3),
        arcface_model_version="buffalo_l",
        preprocessing_version="v1",
        quality_version="heuristic-v1",
        quality_score=0.85,
        quality_components={
            "resolution": 0.9,
            "sharpness": 0.8,
            "detection": 0.95,
            "pitch": 0.9,
        },
        capture_profile={"crop_jpeg_quality": 95},
        interocular_px=32.0,
        yaw_deg=0.0,
        pitch_deg=0.0,
        nearest_person_id="alice",
        nearest_similarity=0.72,
        label_state="labeled",
        label_identity_id="alice",
        label_kind="member",
        label_source="operator_reviewed",
        labeled_by="admin",
        labeled_at=datetime(2026, 11, 21, 12, 5, tzinfo=UTC),
    )
    promoted = await store.promote_candidate(
        cand, approved_by="admin", active_quality_version="heuristic-v1"
    )
    retained_promo_key = enrollment_crop_key("alice", promoted.embedding_id)
    assert retained_promo_key in crops.objects
    # Deleting the original calibration crop leaves the retained enrollment copy intact.
    await crops.delete([cal_key])
    assert retained_promo_key in crops.objects

    # Removing the member deletes every enrollment/alice/* crop and invokes calibration cleanup.
    assert await store.remove_member("alice") is True
    assert not any(k.startswith("enrollment/alice/") for k in crops.objects)
    assert removed_callbacks == ["alice"]


async def test_reembed_writes_parallel_gallery_without_touching_active_gallery() -> None:
    source_repo = InMemoryEnrollmentRepository("buffalo_l")
    target_repo = source_repo.for_recognizer_version(_ADAFACE_PROFILE.recognizer_version)
    crops = _InMemoryCropStore()

    face_a = DetectedFace(
        bbox=[0.0, 0.0, 80.0, 80.0],
        embedding=_unit(10),
        det_score=0.95,
        aligned_crop_bgr=np.full((112, 112, 3), 50, dtype=np.uint8),
        quality=FaceQuality(0.8, 0.8, 0.8, 0.9, 0.9, "heuristic-v1"),
    )
    face_b = DetectedFace(
        bbox=[0.0, 0.0, 80.0, 80.0],
        embedding=_unit(11),
        det_score=0.94,
        aligned_crop_bgr=np.full((112, 112, 3), 150, dtype=np.uint8),
        quality=FaceQuality(0.85, 0.85, 0.85, 0.9, 0.9, "heuristic-v1"),
    )
    store = EnrollmentStore(
        source_repo,
        _QueueFaceEngine([face_a, face_b]),
        DEFAULT_THRESHOLDS,
        crop_store=crops,
    )
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    await store.enroll("alice", "Alice", [dummy])
    await store.enroll("bob", "Bob", [dummy])

    before_source_embs = await source_repo.member_embedding_records()
    before_source_protos = await source_repo.list_prototypes("alice")

    embedder = _DeterministicEmbedder()
    summary = await reembed_gallery(
        source_repo=source_repo,
        target_repo=target_repo,
        crop_reader=crops,
        embedder=embedder,
        source_profile=BUFFALO_L_PROFILE,
        target_profile=_ADAFACE_PROFILE,
    )

    assert summary.total_reembedded == 2
    assert summary.total_skipped == 0
    assert embedder.calls == 2

    # Active buffalo_l gallery is completely untouched.
    after_source_embs = await source_repo.member_embedding_records()
    assert np.allclose(
        after_source_embs["alice"][0].embedding, before_source_embs["alice"][0].embedding
    )
    assert [p.prototype_id for p in await source_repo.list_prototypes("alice")] == [
        p.prototype_id for p in before_source_protos
    ]

    # Parallel adaface gallery has distinct embeddings, centroids, and non-colliding prototype UUIDs.
    target_embs = await target_repo.member_embedding_records()
    assert set(target_embs) == {"alice", "bob"}
    assert target_embs["alice"][0].recognizer_version == _ADAFACE_PROFILE.recognizer_version
    assert not np.allclose(
        target_embs["alice"][0].embedding, before_source_embs["alice"][0].embedding
    )
    target_protos = await target_repo.list_prototypes("alice")
    assert len(target_protos) == 1
    assert target_protos[0].prototype_id != before_source_protos[0].prototype_id

    # Querying each repository matches its own embedding space only.
    base_hits = await source_repo.nearest_centroids(_unit(10), k=1)
    assert base_hits[0].person_id == "alice"
    assert base_hits[0].similarity == pytest.approx(1.0, abs=1e-5)

    ada_hits = await target_repo.nearest_centroids(_unit(1050), k=1)
    assert ada_hits[0].person_id == "alice"
    assert ada_hits[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_reembed_validation_rejects_same_version_wrong_dim_and_template_mismatch() -> None:
    with pytest.raises(ValueError, match="must differ"):
        validate_reembed_profiles(BUFFALO_L_PROFILE, BUFFALO_L_PROFILE)

    dim_128 = RecognizerProfile(
        name="sface_128",
        recognizer_version="sface_128",
        detector_model="face-detector-scrfd",
        recognition_model="face-recognition-sface",
        landmark_2d_model="face-landmark-2d106",
        landmark_3d_model="face-landmark-3d68",
        genderage_model="face-attribute-genderage",
        embedding_dim=128,
        licence="Apache-2.0",
    )
    with pytest.raises(ValueError, match="embedding_dim=128"):
        validate_reembed_profiles(BUFFALO_L_PROFILE, dim_128)

    shifted_tpl = RecognizerProfile(
        name="shifted",
        recognizer_version="shifted",
        detector_model="face-detector-scrfd",
        recognition_model="face-recognition-shifted",
        landmark_2d_model="face-landmark-2d106",
        landmark_3d_model="face-landmark-3d68",
        genderage_model="face-attribute-genderage",
        alignment_template=(
            (30.0, 50.0),
            (70.0, 50.0),
            (50.0, 70.0),
            (40.0, 90.0),
            (60.0, 90.0),
        ),
        licence="test",
    )
    with pytest.raises(ValueError, match="different alignment template"):
        validate_reembed_profiles(BUFFALO_L_PROFILE, shifted_tpl)


@pytest_asyncio.fixture
async def pg_pool() -> AsyncIterator[asyncpg.Pool]:
    if TEST_DSN is None:
        pytest.skip("TEST_DATABASE_URL not set")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM face_prototypes")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


async def test_postgres_reembed_parallel_gallery_and_cascade_delete(
    pg_pool: asyncpg.Pool,
) -> None:
    source_repo = PostgresEnrollmentRepository(pg_pool, recognizer_version="buffalo_l")
    target_repo = PostgresEnrollmentRepository(
        pg_pool, recognizer_version=_ADAFACE_PROFILE.recognizer_version
    )
    crops = _InMemoryCropStore()
    face = DetectedFace(
        bbox=[0.0, 0.0, 80.0, 80.0],
        embedding=_unit(21),
        det_score=0.95,
        aligned_crop_bgr=np.full((112, 112, 3), 75, dtype=np.uint8),
    )
    store = EnrollmentStore(
        source_repo, _QueueFaceEngine([face]), DEFAULT_THRESHOLDS, crop_store=crops
    )
    await store.enroll("alice", "Alice", [np.zeros((160, 160, 3), dtype=np.uint8)])

    summary = await reembed_gallery(
        source_repo=source_repo,
        target_repo=target_repo,
        crop_reader=crops,
        embedder=_DeterministicEmbedder(),
        source_profile=BUFFALO_L_PROFILE,
        target_profile=_ADAFACE_PROFILE,
    )
    assert summary.total_reembedded == 1

    # Both parallel galleries coexist in Postgres without colliding on centroids or prototypes.
    assert await source_repo.embedding_count("alice") == 1
    assert await target_repo.embedding_count("alice") == 1
    assert (await source_repo.nearest_centroids(_unit(21), k=1))[0].similarity == pytest.approx(
        1.0, abs=1e-5
    )
    assert (await target_repo.nearest_centroids(_unit(1075), k=1))[0].similarity == pytest.approx(
        1.0, abs=1e-5
    )

    # Removing the member deletes rows across ALL recognizer versions and clears MinIO crops.
    assert await store.remove_member("alice") is True
    assert await source_repo.embedding_count("alice") == 0
    assert await target_repo.embedding_count("alice") == 0
    assert await target_repo.nearest_centroids(_unit(1075), k=1) == []
    assert not any(k.startswith("enrollment/alice/") for k in crops.objects)


async def test_migration_0008_refuses_while_non_buffalo_rows_exist_and_round_trips(
    pg_pool: asyncpg.Pool,
) -> None:
    alt_repo = PostgresEnrollmentRepository(pg_pool, recognizer_version="adaface_ir50_webface4m")
    await alt_repo.upsert_member("alice", "Alice")
    await alt_repo.insert_embeddings("alice", [_unit(55)])
    await alt_repo.recompute_centroid("alice")

    # Down migration must refuse while non-buffalo_l rows exist.
    with pytest.raises(asyncpg.RaiseError, match="non-buffalo_l"):
        await rollback(pg_pool, steps=1)

    # Once non-buffalo_l rows are removed, 0008 rolls back and re-applies cleanly.
    await alt_repo.remove_member("alice")
    await rollback(pg_pool, steps=1)
    async with pg_pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name = 'centroids' AND column_name = 'recognizer_version'"
            )
            == 0
        )
    await run_migrations(pg_pool)
    assert "0008_recognizer_version" in [s.version for s in await status(pg_pool) if s.applied]
