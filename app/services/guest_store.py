"""Guest image store: uploads unidentified-person frames to MinIO, logs visits to TimescaleDB."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import asyncpg
import cv2
import numpy as np

from app.services.minio_client import MinioClient

logger = logging.getLogger(__name__)


class GuestImageStore:
    """Uploads full-frame images with unidentified persons to MinIO and logs visit metadata."""

    def __init__(self, pool: asyncpg.Pool, minio_client: MinioClient) -> None:
        self._pool = pool
        self._minio = minio_client

    async def save_guest_image(
        self,
        image: np.ndarray,
        guest_count: int = 1,
        frame_index: int = 0,
    ) -> str | None:
        """Encode the frame as JPEG, upload to MinIO, and log to the guest_visits hypertable.

        Args:
            image: Full BGR frame containing one or more unidentified faces.
            guest_count: Number of unidentified faces in this frame.
            frame_index: Frame index within a batch (for filename uniqueness).

        Returns:
            MinIO object name (key), or None if upload failed.
        """
        try:
            # Encode frame as JPEG in memory
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                logger.error("Failed to encode guest image to JPEG")
                return None

            # Object name mirrors old on-disk hierarchy: guests/YYYY-MM-DD/HHMMSS-ffffff_f{idx}_{n}guests.jpg
            now = datetime.now(UTC)
            date_str = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%H%M%S-%f")
            object_name = f"guests/{date_str}/{timestamp}_f{frame_index}_{guest_count}guests.jpg"

            # Upload to MinIO
            self._minio.upload_bytes(buf.tobytes(), object_name, content_type="image/jpeg")

            # Log visit to TimescaleDB
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO guest_visits (guest_count, object_name, frame_index) VALUES ($1, $2, $3)",
                    guest_count,
                    object_name,
                    frame_index,
                )

            logger.info(
                "Saved guest image: %s (%d unidentified, frame %d)",
                object_name,
                guest_count,
                frame_index,
            )
            return object_name

        except Exception:
            logger.exception("Failed to save guest image")
            return None
