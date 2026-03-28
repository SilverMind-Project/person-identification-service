"""Draw bounding boxes and name labels on images for annotated output.

Used when the ``include_annotated_image`` flag is set on identification
requests.  Produces a BGR numpy array with boxes and labels drawn over
a copy of the original frame.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app import config
from app.services.face_engine import IdentifyResult

logger = logging.getLogger(__name__)

# Defaults — overridden from settings.yaml if present
_BOX_COLOR_KNOWN = tuple(config.get("annotation.box_color_known", [0, 200, 0]))
_BOX_COLOR_UNKNOWN = tuple(config.get("annotation.box_color_unknown", [0, 165, 255]))
_TEXT_SCALE = config.get("annotation.text_scale", 0.7)
_TEXT_THICKNESS = config.get("annotation.text_thickness", 2)
_BOX_THICKNESS = config.get("annotation.box_thickness", 2)


def annotate_image(
    image: np.ndarray,
    identities: list[IdentifyResult],
) -> np.ndarray:
    """Draw bounding boxes and name labels on a copy of *image*.

    Known persons are drawn in green, unknown in orange.

    Args:
        image: BGR numpy array (original frame).
        identities: list of identification results with bboxes.

    Returns:
        Annotated BGR numpy array (the original is not mutated).
    """
    annotated = image.copy()

    for ident in identities:
        is_known = not (
            ident.person_id == "unknown" or ident.person_id.startswith("unknown_")
        )
        color = _BOX_COLOR_KNOWN if is_known else _BOX_COLOR_UNKNOWN

        x1, y1, x2, y2 = [int(c) for c in ident.bbox]

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # Label background
        label = f"{ident.name} {ident.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, _TEXT_SCALE, _TEXT_THICKNESS
        )
        label_y = max(y1 - 6, th + 6)
        cv2.rectangle(
            annotated,
            (x1, label_y - th - 6),
            (x1 + tw + 4, label_y + baseline),
            color,
            cv2.FILLED,
        )

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            _TEXT_SCALE,
            (255, 255, 255),
            _TEXT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated
