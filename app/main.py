"""FastAPI application factory for the Person Identification Service."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from pgvector.asyncpg import register_vector

from app import config


async def _init_pool(dsn: str) -> asyncpg.Pool:
    """Create an asyncpg connection pool with pgvector codec registered."""

    async def _init_conn(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    return await asyncpg.create_pool(dsn, init=_init_conn, min_size=1, max_size=5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    log_level = config.get("logging.level", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Person Identification Service")

    # Database pool
    dsn = config.get(
        "database.dsn", "postgresql://cc_user:change-me@localhost:5432/cognitive_companion"
    )
    pool = await _init_pool(dsn)
    logger.info("Database pool created")

    # Ensure schema (idempotent DDL)
    from app.services.enrollment_store import ensure_schema

    await ensure_schema(pool)

    # MinIO client for guest images
    from app.services.minio_client import create_minio_client

    minio_client = create_minio_client()
    logger.info("MinIO client ready, bucket=%s", minio_client.bucket)

    # Face engine (blocking GPU init, run in threadpool)
    from app.services.face_engine import FaceEngine

    face_engine = await asyncio.to_thread(FaceEngine)
    app.state.face_engine = face_engine

    # Enrollment store (asyncpg + pgvector)
    from app.services.enrollment_store import EnrollmentStore

    enrollment_store = EnrollmentStore(pool, face_engine)
    app.state.enrollment_store = enrollment_store

    # Motion detector (stateless, no DB needed)
    from app.services.motion_detector import MotionDetector

    motion_detector = MotionDetector()
    app.state.motion_detector = motion_detector

    # Guest image store (MinIO + TimescaleDB)
    from app.services.guest_store import GuestImageStore

    guest_store = GuestImageStore(pool, minio_client)
    app.state.guest_store = guest_store

    member_count = await enrollment_store.member_count()
    logger.info(
        "Service ready: GPU=%s, enrolled_members=%d",
        face_engine.gpu_available,
        member_count,
    )

    yield

    logger.info("Shutting down Person Identification Service")
    await pool.close()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Person Identification Service",
        version="1.0.0",
        description="Face recognition and motion direction detection for Cognitive Companion",
        lifespan=lifespan,
    )

    from app.routers import enrollment, health, identification, motion

    app.include_router(health.router)
    app.include_router(enrollment.router)
    app.include_router(identification.router)
    app.include_router(motion.router)

    return app


app = create_app()
