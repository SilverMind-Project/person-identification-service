"""M3: Rich face evidence contract — person-id service tests.

Tests for head-pose estimation, three-valued recognition state, and the
extended wire schema.
"""

from __future__ import annotations

import os

import asyncpg
import numpy as np
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from app.db.migrate import run_migrations
from app.models.identification import FaceDetection, IdentifyResponse
from app.services.enrollment_store import EnrollmentStore
from app.services.face_engine import _estimate_head_pose
from app.services.face_models import DetectedFace, IdentifyResult

TEST_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://pid_user:change-me-pid-password@localhost:5432/person_identification",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeFace:
    """Minimal mock of an InsightFace Face (dict subclass) for pose tests."""

    def __init__(self, landmark_3d_68: np.ndarray | None = None):
        self._lm_3d = landmark_3d_68

    @property
    def landmark_3d_68(self):
        return self._lm_3d


def _make_frontal_3d_landmarks() -> np.ndarray:
    """Synthetic frontal-face 3D landmarks (68 points, InsightFace format).

    Points are in pixel space on a 640×640 canvas with a face centred at
    (320, 280). The z coordinate is a small depth offset.
    """
    pts = np.zeros((68, 3), dtype=np.float32)
    # Nose tip (idx 30)
    pts[30] = [320.0, 280.0, 10.0]
    # Nose base right (31), left (35)
    pts[31] = [310.0, 300.0, 5.0]
    pts[35] = [330.0, 300.0, 5.0]
    # Mouth (indices 52, 62, 63, 55, 66, 65)
    pts[52] = [305.0, 360.0, 0.0]
    pts[62] = [320.0, 365.0, 0.0]
    pts[63] = [335.0, 360.0, 0.0]
    pts[55] = [300.0, 370.0, -2.0]
    pts[66] = [320.0, 375.0, -2.0]
    pts[65] = [340.0, 370.0, -2.0]
    # Jaw (indices 1, 9, 8, 7, 15)
    pts[1] = [270.0, 350.0, 0.0]
    pts[9] = [250.0, 380.0, -5.0]
    pts[8] = [320.0, 410.0, -5.0]  # chin
    pts[7] = [390.0, 380.0, -5.0]
    pts[15] = [370.0, 350.0, 0.0]
    # Eyes (indices 40, 42, 27, 39, 37)
    pts[40] = [280.0, 260.0, 15.0]  # right eye outer
    pts[42] = [300.0, 255.0, 15.0]  # right eye inner
    pts[27] = [320.0, 250.0, 15.0]  # bridge
    pts[39] = [340.0, 255.0, 15.0]  # left eye inner
    pts[37] = [360.0, 260.0, 15.0]  # left eye outer
    return pts


def _make_profile_3d_landmarks() -> np.ndarray:
    """Synthetic profile (looking right, ~45°) 3D landmarks."""
    pts = np.zeros((68, 3), dtype=np.float32)
    pts[30] = [340.0, 280.0, 10.0]  # nose tip shifted right
    pts[31] = [330.0, 295.0, 5.0]
    pts[35] = [350.0, 300.0, 3.0]
    pts[52] = [340.0, 360.0, 0.0]
    pts[62] = [350.0, 365.0, 0.0]
    pts[63] = [360.0, 360.0, -2.0]
    pts[55] = [335.0, 368.0, -2.0]
    pts[66] = [350.0, 375.0, -2.0]
    pts[65] = [365.0, 368.0, -4.0]
    pts[1] = [300.0, 345.0, 0.0]
    pts[9] = [290.0, 375.0, -5.0]
    pts[8] = [340.0, 408.0, -5.0]
    pts[7] = [390.0, 375.0, -8.0]
    pts[15] = [380.0, 345.0, -2.0]
    pts[40] = [305.0, 258.0, 15.0]
    pts[42] = [320.0, 253.0, 13.0]
    pts[27] = [335.0, 248.0, 10.0]
    pts[39] = [350.0, 255.0, 8.0]
    pts[37] = [365.0, 260.0, 5.0]
    return pts


def _alice_embedding() -> np.ndarray:
    rng = np.random.RandomState(42)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _bob_embedding() -> np.ndarray:
    rng = np.random.RandomState(99)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


def _embedding_at_similarity(ref: np.ndarray, target_sim: float) -> np.ndarray:
    """Return an embedding at approximately *target_sim* cosine similarity to *ref*."""
    rng = np.random.RandomState(7)
    noise = rng.randn(512).astype(np.float32)
    parallel = np.dot(noise, ref) * ref
    perp = noise - parallel
    if np.linalg.norm(perp) < 1e-8:
        perp = rng.randn(512).astype(np.float32)
    perp = perp / np.linalg.norm(perp)
    sim = max(-1.0, min(1.0, target_sim))
    ortho_weight = np.sqrt(max(0.0, 1.0 - sim * sim))
    result = sim * ref + ortho_weight * perp
    return (result / np.linalg.norm(result)).astype(np.float32)


async def _can_connect() -> bool:
    try:
        conn = await asyncpg.connect(TEST_DSN, timeout=5)
        await conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# A1: Head pose estimation (no DB required)
# ---------------------------------------------------------------------------


class TestHeadPoseEstimation:
    def test_frontal_pose_returns_finite_values(self):
        """Frontal face yields finite (non-NaN, non-inf) pose angles."""
        lm = _make_frontal_3d_landmarks()
        face = _FakeFace(landmark_3d_68=lm)
        yaw, pitch, roll = _estimate_head_pose(face)
        assert np.isfinite(yaw)
        assert np.isfinite(pitch)
        assert np.isfinite(roll)

    def test_profile_pose_returns_finite_values(self):
        """Profile face also yields finite pose angles (different geometry)."""
        lm = _make_profile_3d_landmarks()
        face = _FakeFace(landmark_3d_68=lm)
        yaw, pitch, roll = _estimate_head_pose(face)
        assert np.isfinite(yaw)
        assert np.isfinite(pitch)
        assert np.isfinite(roll)

    def test_missing_landmarks_returns_zero(self):
        """When 3D landmarks are absent, pose returns (0, 0, 0)."""
        face = _FakeFace(landmark_3d_68=None)
        yaw, pitch, roll = _estimate_head_pose(face)
        assert yaw == 0.0
        assert pitch == 0.0
        assert roll == 0.0

    def test_different_geometry_yields_different_pose(self):
        """Two distinct landmark configurations produce different pose angles."""
        lm_a = _make_frontal_3d_landmarks()
        lm_b = _make_profile_3d_landmarks()
        ya, pa, ra = _estimate_head_pose(_FakeFace(landmark_3d_68=lm_a))
        yb, pb, rb = _estimate_head_pose(_FakeFace(landmark_3d_68=lm_b))
        # At least one angle component should differ.
        different = (abs(ya - yb) > 0.5) or (abs(pa - pb) > 0.5) or (abs(ra - rb) > 0.5)
        assert different, f"Expected different poses, got ({ya:.1f},{pa:.1f},{ra:.1f}) vs ({yb:.1f},{pb:.1f},{rb:.1f})"


# ---------------------------------------------------------------------------
# A2: Recognition state (three-valued, needs DB pool)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def recognition_pool():
    """Test pool with clean schema, skipped if DB unavailable."""
    if not await _can_connect():
        pytest.skip("Database not available")

    async def _init(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(TEST_DSN, init=_init, min_size=1, max_size=3)
    await run_migrations(pool)
    yield pool
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM guest_visits")
        await conn.execute("DELETE FROM embeddings")
        await conn.execute("DELETE FROM centroids")
        await conn.execute("DELETE FROM members")
    await pool.close()


@pytest_asyncio.fixture
async def seeded_store(recognition_pool):
    """Store with alice and bob seeded in the gallery."""
    # Insert alice and bob into members + centroids.
    async with recognition_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO members (person_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            "alice", "Alice",
        )
        await conn.execute(
            "INSERT INTO centroids (person_id, centroid, updated_at) VALUES ($1, $2, now()) "
            "ON CONFLICT (person_id) DO UPDATE SET centroid = excluded.centroid",
            "alice", _alice_embedding().tolist(),
        )
        await conn.execute(
            "INSERT INTO members (person_id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            "bob", "Bob",
        )
        await conn.execute(
            "INSERT INTO centroids (person_id, centroid, updated_at) VALUES ($1, $2, now()) "
            "ON CONFLICT (person_id) DO UPDATE SET centroid = excluded.centroid",
            "bob", _bob_embedding().tolist(),
        )

    class _NoOpFaceEngine:
        def detect_faces(self, image):
            return []

    yield EnrollmentStore(recognition_pool, _NoOpFaceEngine())


class TestRecognitionState:
    """Verify the three recognition states (recognized / candidate / unrecognized)."""

    @pytest.mark.asyncio
    async def test_recognized_state(self, seeded_store):
        """Similarity >= threshold → recognized, person_id = best_candidate_id."""
        identity = await seeded_store.identify(_alice_embedding())
        assert identity.recognition_state == "recognized"
        assert identity.person_id == "alice"
        assert identity.best_candidate_id == "alice"
        assert identity.similarity > 0.9

    @pytest.mark.asyncio
    async def test_candidate_state(self, seeded_store):
        """Similarity in the grey zone → candidate, person_id = 'unknown'."""
        grey_emb = _embedding_at_similarity(_alice_embedding(), target_sim=0.35)
        identity = await seeded_store.identify(grey_emb)
        assert identity.recognition_state == "candidate"
        assert identity.person_id == "unknown"
        assert identity.best_candidate_id == "alice"
        assert identity.similarity < 0.4

    @pytest.mark.asyncio
    async def test_unrecognized_state(self, seeded_store):
        """Similarity below unknown_threshold → unrecognized, person_id = 'unknown'."""
        far_emb = _embedding_at_similarity(_alice_embedding(), target_sim=0.15)
        identity = await seeded_store.identify(far_emb)
        assert identity.recognition_state == "unrecognized"
        assert identity.person_id == "unknown"
        assert identity.best_candidate_id == "alice"
        assert identity.similarity < 0.25

    @pytest.mark.asyncio
    async def test_best_candidate_populated_even_when_unrecognized(self, seeded_store):
        """Even when unrecognized, best_candidate_id carries the nearest centroid."""
        far_emb = _embedding_at_similarity(_bob_embedding(), target_sim=0.12)
        identity = await seeded_store.identify(far_emb)
        assert identity.best_candidate_id == "bob"
        assert identity.recognition_state == "unrecognized"


# ---------------------------------------------------------------------------
# A3: Wire schema (FaceDetection model, no DB required)
# ---------------------------------------------------------------------------


class TestFaceDetectionSchema:
    """Verify the extended FaceDetection Pydantic model."""

    def test_face_detection_defaults(self):
        """All new fields have safe defaults so older clients parse unchanged."""
        fd = FaceDetection(
            person_id="alice",
            name="Alice",
            confidence=0.85,
            bbox=[10, 20, 100, 120],
        )
        assert fd.recognition_state == "recognized"
        assert fd.best_candidate_id is None
        assert fd.similarity == 0.0
        assert fd.yaw_deg == 0.0
        assert fd.pitch_deg == 0.0
        assert fd.roll_deg == 0.0
        assert fd.det_score == 0.0

    def test_face_detection_new_fields_populated(self):
        """New fields can be explicitly set."""
        fd = FaceDetection(
            person_id="unknown",
            name="Guest",
            confidence=0.33,
            bbox=[50, 40, 150, 180],
            recognition_state="candidate",
            best_candidate_id="alice",
            similarity=0.33,
            yaw_deg=35.0,
            pitch_deg=-5.0,
            roll_deg=2.0,
            det_score=0.82,
        )
        assert fd.recognition_state == "candidate"
        assert fd.best_candidate_id == "alice"
        assert fd.similarity == 0.33
        assert fd.yaw_deg == 35.0
        assert fd.pitch_deg == -5.0
        assert fd.roll_deg == 2.0
        assert fd.det_score == 0.82

    def test_response_serialization_includes_new_fields(self):
        """IdentifyResponse serializes the new fields in JSON output."""
        fd = FaceDetection(
            person_id="alice",
            name="Alice",
            confidence=0.88,
            bbox=[10, 20, 100, 120],
            recognition_state="recognized",
            best_candidate_id="alice",
            similarity=0.88,
            yaw_deg=5.0,
            pitch_deg=2.0,
            roll_deg=-1.0,
            det_score=0.95,
        )
        response = IdentifyResponse(faces=[fd])
        data = response.model_dump()
        face_data = data["faces"][0]
        assert face_data["recognition_state"] == "recognized"
        assert face_data["best_candidate_id"] == "alice"
        assert face_data["similarity"] == 0.88
        assert face_data["yaw_deg"] == 5.0
        assert face_data["pitch_deg"] == 2.0
        assert face_data["roll_deg"] == -1.0
        assert face_data["det_score"] == 0.95


# ---------------------------------------------------------------------------
# A4: IdentifyResult carries detection-level fields (no DB required)
# ---------------------------------------------------------------------------


class TestIdentifyResultCarriesDetectionFields:
    def test_identify_result_defaults(self):
        """IdentifyResult has pose and det_score with safe defaults for backward compat."""
        ir = IdentifyResult(
            person_id="alice",
            name="Alice",
            confidence=0.9,
            bbox=[10, 20, 100, 120],
        )
        assert ir.yaw_deg == 0.0
        assert ir.pitch_deg == 0.0
        assert ir.roll_deg == 0.0
        assert ir.det_score == 0.0
        assert ir.best_candidate_id is None
        assert ir.similarity == 0.0
        assert ir.recognition_state == "recognized"

    def test_detected_face_pose_defaults(self):
        """DetectedFace has pose fields with defaults for backward compat."""
        df = DetectedFace(
            bbox=[10, 20, 100, 120],
            embedding=np.zeros(512, dtype=np.float32),
            det_score=0.9,
        )
        assert df.yaw_deg == 0.0
        assert df.pitch_deg == 0.0
        assert df.roll_deg == 0.0
