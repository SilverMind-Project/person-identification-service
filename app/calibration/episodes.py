"""Capture episodes: the unit calibration splits household data by.

An episode is a maximal run of captures with the same label (a household identity, or "not
enrolled") on one camera, with no gap between consecutive captures of ``gap_s`` or more. Adjacent
frames of one walk past a camera are nearly identical, so they must land in the same split;
splitting by frame would leak a person's walk into both fit and holdout. Stdlib only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

DEFAULT_EPISODE_GAP_S = 300.0
NOT_ENROLLED = "not_enrolled"


@dataclass(frozen=True)
class EpisodeInput:
    candidate_id: str
    captured_at: datetime
    camera_id: str
    # The labeled identity, or NOT_ENROLLED for a person who is not a household member.
    label: str


def episode_label(label_kind: str | None, label_identity_id: str | None) -> str:
    """A member's identity, or NOT_ENROLLED for anyone else."""
    if label_kind == "member" and label_identity_id is not None:
        return label_identity_id
    return NOT_ENROLLED


def assign_episodes(
    captures: Iterable[EpisodeInput], gap_s: float = DEFAULT_EPISODE_GAP_S
) -> dict[str, str]:
    """Map each candidate id to its episode id (``label|camera|start time``), deterministically."""
    by_stream: dict[tuple[str, str], list[EpisodeInput]] = {}
    for capture in captures:
        by_stream.setdefault((capture.label, capture.camera_id), []).append(capture)
    episodes: dict[str, str] = {}
    for (label, camera_id), stream in by_stream.items():
        stream.sort(key=lambda c: (c.captured_at, c.candidate_id))
        episode_start: datetime | None = None
        previous: datetime | None = None
        for capture in stream:
            if previous is None or (capture.captured_at - previous).total_seconds() >= gap_s:
                episode_start = capture.captured_at
            assert episode_start is not None
            episodes[capture.candidate_id] = f"{label}|{camera_id}|{episode_start.isoformat()}"
            previous = capture.captured_at
    return episodes
