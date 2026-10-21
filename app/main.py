"""FastAPI application factory for the Person Identification Service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import asyncpg
from fastapi import FastAPI
from pgvector.asyncpg import register_vector
from triton_shared.client import TritonGrpcClient

from app import config
from app.calibration.evaluator import CalibrationEvaluator
from app.db.migrate import run_migrations
from app.routers import calibration, enrollment, health, identification, motion, visitors
from app.services.calibration_candidates import PostgresCalibrationCandidateRepository
from app.services.calibration_capture import (
    CalibrationCaptureService,
    CapturePolicy,
    CaptureVersions,
    MinioCropStore,
)
from app.services.enrollment_repository import PostgresEnrollmentRepository
from app.services.enrollment_store import EnrollmentStore
from app.services.face_engine import FaceEngine
from app.services.face_quality import build_face_quality_scorer
from app.services.guest_store import GuestImageStore
from app.services.match_policy import MatchThresholds
from app.services.minio_client import create_minio_client
from app.services.motion_detector import MotionDetector
from app.services.visitor_store import VisitorStore

_RETENTION_INTERVAL_S = 24 * 60 * 60


async def _retention_loop(
    visitor_store: VisitorStore, calibration_capture: CalibrationCaptureService
) -> None:
    """Daily loop that purges expired unnamed visitor clusters and expired calibration captures."""
    logger = logging.getLogger(__name__)
    while True:
        try:
            await asyncio.sleep(_RETENTION_INTERVAL_S)
            await visitor_store.purge_expired()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Visitor cluster retention job failed")
        try:
            purged = await calibration_capture.purge_expired(datetime.now(UTC))
            logger.info("calibration_candidates_purged count=%d", purged)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Calibration capture retention job failed")


def _capture_policy() -> CapturePolicy:
    section = config.get("calibration_capture", {}) or {}
    return CapturePolicy(
        enabled=bool(section.get("enabled", False)),
        per_bucket_per_day=int(section.get("per_bucket_per_day", 40)),
        det_threshold=float(config.get("face_engine.det_threshold", 0.6)),
        unlabeled_retention_days=int(section.get("unlabeled_retention_days", 30)),
        queue_size=int(section.get("queue_size", 64)),
    )


async def _init_pool(dsn: str) -> asyncpg.Pool:
    """Create an asyncpg connection pool with pgvector codec registered."""

    async def _init_conn(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    return await asyncpg.create_pool(dsn, init=_init_conn, min_size=1, max_size=5)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    log_level = config.get("logging.level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Person Identification Service")

    # Configuration errors fail before any connection is opened.
    quality_scorer = build_face_quality_scorer(config.get("face_quality", {}))
    capture_policy = _capture_policy()

    # Database pool
    dsn = config.get("database.dsn")
    if not dsn:
        raise RuntimeError(
            "database.dsn is not configured. "
            "Set DATABASE_URL in the environment (e.g. postgresql://user:pass@host:5432/person_identification)."
        )
    pool = await _init_pool(dsn)
    logger.info("Database pool created")

    triton_client = None
    retention_task: asyncio.Task | None = None
    calibration_capture: CalibrationCaptureService | None = None
    try:
        # Run migrations (idempotent and safe for fresh or existing databases).
        applied = await run_migrations(pool)
        if applied:
            logger.info("Database migrations applied count=%d", applied)

        minio_client = create_minio_client()
        logger.info("MinIO client ready, bucket=%s", minio_client.bucket)

        triton_url = str(config.get("face_engine.triton_url", "triton:8701"))
        triton_timeout_ms = int(config.get("face_engine.triton_timeout_ms", 30_000))
        triton_client = TritonGrpcClient(triton_url, timeout_ms=triton_timeout_ms)
        await triton_client.__aenter__()
        face_engine = FaceEngine(triton_client, quality_scorer)
        await face_engine.validate_models()
        app.state.face_engine = face_engine
        app.state.triton_client = triton_client

        enrollment_store = EnrollmentStore(
            PostgresEnrollmentRepository(pool),
            face_engine,
            MatchThresholds(
                recognized_min_similarity=float(config.get("recognition.threshold", 0.4)),
                candidate_min_similarity=float(config.get("recognition.unknown_threshold", 0.25)),
            ),
        )
        app.state.enrollment_store = enrollment_store

        app.state.motion_detector = MotionDetector()

        app.state.guest_store = GuestImageStore(pool, minio_client)

        visitor_store = VisitorStore(pool, minio_client)
        app.state.visitor_store = visitor_store

        calibration_candidates = PostgresCalibrationCandidateRepository(pool)
        app.state.calibration_candidates = calibration_candidates
        calibration_capture = CalibrationCaptureService(
            calibration_candidates,
            MinioCropStore(minio_client),
            capture_policy,
            CaptureVersions(
                arcface_model_version=str(config.get("calibration.arcface_model_version", "")),
                preprocessing_version=str(config.get("calibration.preprocessing_version", "")),
            ),
        )
        await calibration_capture.start()
        app.state.calibration_capture = calibration_capture
        logger.info("Calibration capture enabled=%s", calibration_capture.enabled)

        retention_task = asyncio.create_task(_retention_loop(visitor_store, calibration_capture))

        calibration_evaluator = CalibrationEvaluator.from_artifact_path(
            config.get("calibration.artifact_path", "") or None,
            expected_arcface_model_version=str(config.get("calibration.arcface_model_version", "")),
            expected_model_profile=str(face_engine.model_profile),
            expected_preprocessing_version=str(config.get("calibration.preprocessing_version", "")),
            expected_quality_version=quality_scorer.version,
        )
        app.state.calibration_evaluator = calibration_evaluator
        logger.info(
            "Calibration evaluator status=%s artifact_version=%s",
            calibration_evaluator.health(),
            calibration_evaluator.artifact_version(),
        )

        member_count = await enrollment_store.member_count()
        logger.info(
            "Service ready: inference_backend=triton endpoint=%s profile=%s enrolled_members=%d",
            face_engine.endpoint,
            face_engine.model_profile,
            member_count,
        )
        yield
    finally:
        logger.info("Shutting down Person Identification Service")
        if retention_task is not None:
            retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await retention_task
        if calibration_capture is not None:
            await calibration_capture.stop()
        if triton_client is not None:
            await triton_client.__aexit__(None, None, None)
        await pool.close()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Person Identification Service",
        version="1.0.0",
        description="Face recognition and motion direction detection for Cognitive Companion",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(enrollment.router)
    app.include_router(visitors.router)
    app.include_router(calibration.router)
    app.include_router(identification.router)
    app.include_router(motion.router)

    return app


app = create_app()
