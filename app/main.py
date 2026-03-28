"""FastAPI application factory for the Person Identification Service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    log_level = config.get("logging.level", "INFO")
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting Person Identification Service")

    # Initialize face engine (loads InsightFace model)
    from app.services.face_engine import FaceEngine

    face_engine = FaceEngine()
    app.state.face_engine = face_engine

    # Initialize enrollment store (loads centroids into memory)
    from app.services.enrollment_store import EnrollmentStore

    enrollment_store = EnrollmentStore(face_engine)
    app.state.enrollment_store = enrollment_store

    # Initialize motion detector
    from app.services.motion_detector import MotionDetector

    motion_detector = MotionDetector()
    app.state.motion_detector = motion_detector

    # Initialize guest image store
    from app.services.guest_store import GuestImageStore

    guest_store = GuestImageStore()
    app.state.guest_store = guest_store

    logger.info(
        "Service ready: GPU=%s, enrolled_members=%d",
        face_engine.gpu_available,
        enrollment_store.member_count,
    )

    yield

    logger.info("Shutting down Person Identification Service")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Person Identification Service",
        version="1.0.0",
        description="Face recognition and motion direction detection for Cognitive Companion",
        lifespan=lifespan,
    )

    from app.routers import enrollment, identification, motion, health

    app.include_router(health.router)
    app.include_router(enrollment.router)
    app.include_router(identification.router)
    app.include_router(motion.router)

    return app


app = create_app()
