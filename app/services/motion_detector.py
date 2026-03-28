"""Motion direction detection via cross-frame centroid tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from app import config
from app.services.face_engine import DetectedFace, FaceEngine, IdentifyResult

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class PersonTrackResult:
    track_id: int
    person_id: str
    name: str
    direction: str  # left-to-right, right-to-left, towards-camera, away-from-camera, stationary
    confidence: float
    trajectory: list[TrackPoint]


class MotionDetector:
    """Detects direction of motion for persons across a sequence of frames."""

    def __init__(self) -> None:
        self._min_disp_frac = config.get("motion.min_displacement_fraction", 0.05)
        self._cross_frame_sim = config.get("motion.cross_frame_similarity", 0.5)

    def detect_direction(
        self,
        frame_shapes: list[tuple[int, int]],  # (height, width) per frame
        frame_faces: list[list[DetectedFace]],
        frame_identities: list[list[IdentifyResult]],
    ) -> list[PersonTrackResult]:
        """Compute motion direction for each tracked person across frames.

        Args:
            frame_shapes: (height, width) of each frame.
            frame_faces: Detected faces per frame.
            frame_identities: Identification results per frame (parallel to frame_faces).

        Returns:
            List of PersonTrackResult with direction and trajectory.
        """
        if len(frame_faces) < 2:
            return []

        # Build person tracks: person_id -> list of (frame_idx, identity, face)
        tracks: dict[str, list[tuple[int, IdentifyResult, DetectedFace]]] = {}

        for frame_idx, (faces, identities) in enumerate(zip(frame_faces, frame_identities)):
            for face, identity in zip(faces, identities):
                pid = identity.person_id
                tracks.setdefault(pid, []).append((frame_idx, identity, face))

        # For unknown faces, try to link them across frames by embedding similarity
        unknown_entries = tracks.pop("unknown", [])
        if unknown_entries:
            unknown_tracks = self._link_unknowns(unknown_entries)
            tracks.update(unknown_tracks)

        results: list[PersonTrackResult] = []
        track_id = 0

        for person_id, entries in tracks.items():
            if len(entries) < 2:
                continue

            entries.sort(key=lambda e: e[0])  # sort by frame index

            trajectory: list[TrackPoint] = []
            for _, identity, face in entries:
                x1, y1, x2, y2 = face.bbox
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                trajectory.append(TrackPoint(cx=cx, cy=cy, width=w, height=h))

            direction, dir_confidence = self._classify_direction(
                trajectory,
                frame_shapes[entries[0][0]],
            )

            avg_confidence = sum(e[1].confidence for e in entries) / len(entries)
            name = entries[0][1].name

            results.append(
                PersonTrackResult(
                    track_id=track_id,
                    person_id=person_id,
                    name=name,
                    direction=direction,
                    confidence=min(avg_confidence, dir_confidence),
                    trajectory=trajectory,
                )
            )
            track_id += 1

        return results

    def _classify_direction(
        self,
        trajectory: list[TrackPoint],
        frame_shape: tuple[int, int],
    ) -> tuple[str, float]:
        """Classify movement direction from a trajectory.

        Returns (direction_string, confidence).
        """
        if len(trajectory) < 2:
            return "stationary", 0.0

        frame_h, frame_w = frame_shape

        first = trajectory[0]
        last = trajectory[-1]

        # Horizontal displacement
        dx = last.cx - first.cx
        dx_frac = abs(dx) / frame_w if frame_w > 0 else 0.0

        # Depth proxy: change in face area
        first_area = first.width * first.height
        last_area = last.width * last.height
        avg_area = (first_area + last_area) / 2 if (first_area + last_area) > 0 else 1.0
        area_change_frac = (last_area - first_area) / avg_area

        horizontal_significant = dx_frac >= self._min_disp_frac
        depth_significant = abs(area_change_frac) >= 0.15  # 15% area change

        # Determine dominant direction
        if horizontal_significant and depth_significant:
            # Both significant: pick the dominant one, but report both
            if dx_frac > abs(area_change_frac):
                direction = "left-to-right" if dx > 0 else "right-to-left"
                confidence = min(1.0, dx_frac / self._min_disp_frac * 0.5)
            else:
                direction = "towards-camera" if area_change_frac > 0 else "away-from-camera"
                confidence = min(1.0, abs(area_change_frac))
        elif horizontal_significant:
            direction = "left-to-right" if dx > 0 else "right-to-left"
            confidence = min(1.0, dx_frac / self._min_disp_frac * 0.5)
        elif depth_significant:
            direction = "towards-camera" if area_change_frac > 0 else "away-from-camera"
            confidence = min(1.0, abs(area_change_frac))
        else:
            direction = "stationary"
            confidence = 1.0 - max(dx_frac / self._min_disp_frac, abs(area_change_frac) / 0.15)
            confidence = max(0.0, confidence)

        return direction, confidence

    def _link_unknowns(
        self,
        entries: list[tuple[int, IdentifyResult, DetectedFace]],
    ) -> dict[str, list[tuple[int, IdentifyResult, DetectedFace]]]:
        """Link unknown face detections across frames by embedding similarity.

        Returns tracks keyed by synthetic person IDs like "unknown_0", "unknown_1".
        """
        if not entries:
            return {}

        # Sort by frame index
        entries.sort(key=lambda e: e[0])

        tracks: list[list[tuple[int, IdentifyResult, DetectedFace]]] = []

        for entry in entries:
            frame_idx, identity, face = entry
            matched = False
            for track in tracks:
                # Compare with the last face in this track
                last_face = track[-1][2]
                sim = FaceEngine.compute_similarity(face.embedding, last_face.embedding)
                if sim >= self._cross_frame_sim:
                    track.append(entry)
                    matched = True
                    break
            if not matched:
                tracks.append([entry])

        return {
            f"unknown_{i}": track
            for i, track in enumerate(tracks)
            if len(track) >= 2
        }
