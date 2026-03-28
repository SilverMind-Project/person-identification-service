"""Guest image store: saves full images containing unidentified persons to disk."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app import config

logger = logging.getLogger(__name__)


class GuestImageStore:
    """Saves full frame images when unidentified (guest) persons are detected."""

    def __init__(self) -> None:
        self._guest_dir = Path(config.get("storage.guest_images_dir", "data/guests"))
        self._guest_dir.mkdir(parents=True, exist_ok=True)

    def save_guest_image(
        self,
        image: np.ndarray,
        guest_count: int = 1,
        frame_index: int = 0,
    ) -> str | None:
        """Save the full frame image when unidentified guests are present.

        Args:
            image: Full BGR frame containing one or more unidentified faces.
            guest_count: Number of unidentified faces in this frame.
            frame_index: Frame index within a batch (for filename uniqueness).

        Returns:
            Path to the saved image, or None if saving failed.
        """
        try:
            # Organize by date: data/guests/2026-03-23/
            now = datetime.now(timezone.utc)
            date_dir = self._guest_dir / now.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)

            # Filename: {timestamp}_f{frame_index}_{guest_count}guests.jpg
            timestamp = now.strftime("%H%M%S-%f")
            filename = f"{timestamp}_f{frame_index}_{guest_count}guests.jpg"
            path = date_dir / filename

            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.info("Saved guest image: %s (%d unidentified)", path, guest_count)
            return str(path)

        except Exception:
            logger.exception("Failed to save guest image")
            return None
