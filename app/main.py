"""FastAPI application factory for the Person Identification Service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from pgvector.asyncpg import register_vector
from triton_shared.client import TritonGrpcClient

from app import config
from app.db.migrate import run_migrations
from app.routers import enrollment, health, identification, motion
from app.services.enrollment_store import EnrollmentStore
from app.services.face_engine import FaceEngine
from app.services.guest_store import GuestImageStore
from app.services.minio_client import create_minio_client
from app.services.motion_detector import MotionDetector


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
        face_engine = FaceEngine(triton_client)
        await face_engine.validate_models()
        app.state.face_engine = face_engine
        app.state.triton_client = triton_client

        enrollment_store = EnrollmentStore(pool, face_engine)
        app.state.enrollment_store = enrollment_store

        app.state.motion_detector = MotionDetector()

        app.state.guest_store = GuestImageStore(pool, minio_client)

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
    app.include_router(identification.router)
    app.include_router(motion.router)

    return app


app = create_app()
