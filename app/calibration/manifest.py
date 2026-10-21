"""Calibration manifest v2: labeled faces turned into probe-versus-gallery score records.

What is calibrated is P(probe belongs to the matched identity | similarity to that identity's
gallery representation, probe quality), so records are built exactly that way: every usable probe
is scored against every enrolled identity's centroid. A probe of identity A against A is genuine
(label 1); against B it is a closed-set impostor; a probe of someone not enrolled is an open-set
impostor against every identity (label 0 only). Visitors and relatives are the realistic
false-match risk, so open-set impostors matter most.

Leave-episode-out: a probe's own identity's centroid excludes gallery embeddings derived from the
probe's episode, so adjacent frames of the same walk never inflate genuine scores. Enrollment
embeddings carry no capture provenance today, so the exclusion applies once promoted captures
record their source candidate.

External faces (an optional research set) go through the same face engine and quality scorer as
live faces; their records are scored against their own identities' leave-one-out centroids and are
marked ``dataset = external`` so the fitter can split them by identity and never gate on them.
Numpy only; offline.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from app.calibration.episodes import EpisodeInput, assign_episodes, episode_label
from app.services.calibration_candidates import CalibrationCandidate
from app.services.face_models import DetectedFace

Dataset = Literal["household", "external"]
QUALITY_FIELDS = ("quality_score", "resolution", "sharpness", "detection", "pitch")


class ManifestError(ValueError):
    """The labeled data cannot form a consistent manifest; nothing was written."""


@dataclass(frozen=True)
class ScoreRecord:
    probe_id: str
    episode_id: str
    identity_id: str  # the gallery identity scored against
    probe_identity_id: str | None  # the probe's label; None for a person who is not enrolled
    label: int
    similarity: float
    quality_score: float
    resolution: float
    sharpness: float
    detection: float
    pitch: float
    camera_id: str
    label_source: str
    dataset: Dataset


@dataclass(frozen=True)
class GalleryEmbedding:
    embedding: np.ndarray
    # The calibration candidate the embedding was promoted from, when it was; None for photos.
    source_candidate_id: str | None = None


@dataclass
class ManifestMeta:
    records: int = 0
    genuine: int = 0
    impostor: int = 0
    open_set_probes: int = 0
    probes_by_identity: dict[str, int] = field(default_factory=dict)
    skipped_not_a_face: int = 0
    skipped_identity_not_enrolled: int = 0
    quality_version: str = ""
    arcface_model_version: str = ""
    preprocessing_version: str = ""
    capture_profile: dict[str, float] = field(default_factory=dict)
    external_datasets: list[dict[str, Any]] = field(default_factory=list)
    dataset_hash: str = ""


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float64)


def _centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray | None:
    if not embeddings:
        return None
    return _unit(np.mean(np.asarray(embeddings, dtype=np.float64), axis=0))


def _single(values: Iterable[Any], what: str) -> Any:
    distinct = {json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for v in values}
    if len(distinct) > 1:
        raise ManifestError(
            f"labeled captures mix {what} values {sorted(map(str, distinct))}; fit one at a time"
        )
    return next(iter(distinct), None)


def household_records(
    candidates: Sequence[CalibrationCandidate],
    gallery: Mapping[str, Sequence[GalleryEmbedding]],
    *,
    episode_gap_s: float,
    meta: ManifestMeta,
) -> list[ScoreRecord]:
    """Score records for labeled household captures against every enrolled identity."""
    usable: list[CalibrationCandidate] = []
    for c in candidates:
        if c.label_state != "labeled" or c.label_kind is None:
            continue
        if c.label_kind == "not_a_face":
            meta.skipped_not_a_face += 1
            continue
        if c.label_kind == "member" and c.label_identity_id not in gallery:
            meta.skipped_identity_not_enrolled += 1
            continue
        usable.append(c)
    if not usable:
        return []

    meta.quality_version = _single((c.quality_version for c in usable), "quality_version")
    meta.arcface_model_version = _single(
        (c.arcface_model_version for c in usable), "arcface_model_version"
    )
    meta.preprocessing_version = _single(
        (c.preprocessing_version for c in usable), "preprocessing_version"
    )
    profile = _single((dict(c.capture_profile) for c in usable), "capture_profile")
    meta.capture_profile = json.loads(profile) if profile else {}

    episodes = assign_episodes(
        (
            EpisodeInput(
                candidate_id=c.candidate_id,
                captured_at=c.captured_at,
                camera_id=c.camera_id,
                label=episode_label(c.label_kind, c.label_identity_id),
            )
            for c in usable
        ),
        gap_s=episode_gap_s,
    )
    episode_members: dict[str, set[str]] = {}
    for candidate_id, episode in episodes.items():
        episode_members.setdefault(episode, set()).add(candidate_id)
    full_centroids = {
        identity: _centroid([g.embedding for g in embeddings])
        for identity, embeddings in gallery.items()
    }

    records: list[ScoreRecord] = []
    for c in usable:
        probe_identity = c.label_identity_id if c.label_kind == "member" else None
        episode = episodes[c.candidate_id]
        probe = _unit(np.asarray(c.embedding, dtype=np.float64))
        for identity in sorted(gallery):
            centroid = full_centroids[identity]
            if identity == probe_identity:
                same_episode = episode_members[episode]
                centroid = _centroid(
                    [
                        g.embedding
                        for g in gallery[identity]
                        if g.source_candidate_id not in same_episode
                    ]
                )
            if centroid is None:
                continue
            records.append(
                ScoreRecord(
                    probe_id=c.candidate_id,
                    episode_id=episode,
                    identity_id=identity,
                    probe_identity_id=probe_identity,
                    label=int(identity == probe_identity),
                    similarity=float(np.dot(probe, centroid)),
                    quality_score=c.quality_score,
                    resolution=float(c.quality_components["resolution"]),
                    sharpness=float(c.quality_components["sharpness"]),
                    detection=float(c.quality_components["detection"]),
                    pitch=float(c.quality_components["pitch"]),
                    camera_id=c.camera_id,
                    label_source=str(c.label_source),
                    dataset="household",
                )
            )
    meta.open_set_probes += sum(1 for c in usable if c.label_kind == "not_enrolled")
    for c in usable:
        if c.label_identity_id is not None:
            meta.probes_by_identity[c.label_identity_id] = (
                meta.probes_by_identity.get(c.label_identity_id, 0) + 1
            )
    return records


# --- external research sets ---------------------------------------------------------------


class FaceDetector(Protocol):
    async def detect_faces(self, image: np.ndarray) -> list[DetectedFace]: ...


@dataclass(frozen=True)
class ExternalFace:
    probe_id: str
    identity_id: str
    embedding: np.ndarray
    face: DetectedFace


def load_external_listing(path: Path) -> tuple[list[tuple[Path, str]], dict[str, str]]:
    """Read an external set: a JSONL of ``{"image", "identity_id"}`` plus ``<path>.licence.json``.

    The sidecar must name the dataset, its version, and its licence text; a set without one is
    refused. Image paths are relative to the listing.
    """
    sidecar = path.with_name(path.name + ".licence.json")
    if not sidecar.is_file():
        raise ManifestError(f"external set {path} has no licence sidecar {sidecar.name}")
    info = json.loads(sidecar.read_text())
    if not all(str(info.get(key, "")).strip() for key in ("name", "version", "licence")):
        raise ManifestError("external licence sidecar must give name, version and licence")
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            entries.append((path.parent / item["image"], str(item["identity_id"])))
    return entries, {k: str(info[k]) for k in ("name", "version", "licence")}


async def detect_external_faces(
    entries: Sequence[tuple[Path, str]], detector: FaceDetector
) -> tuple[list[ExternalFace], int]:
    """Run external images through the live face engine; returns faces and the count dropped."""
    faces: list[ExternalFace] = []
    dropped = 0
    for index, (image_path, identity) in enumerate(entries):
        image = cv2.imread(str(image_path))
        detected = await detector.detect_faces(image) if image is not None else []
        scored = [f for f in detected if f.quality is not None]
        if not scored:
            dropped += 1
            continue
        best = max(scored, key=lambda f: f.geometry.face_width_px * f.geometry.face_height_px)
        faces.append(ExternalFace(f"external:{index}", identity, best.embedding, best))
    return faces, dropped


def external_records(faces: Sequence[ExternalFace]) -> list[ScoreRecord]:
    """Each external probe against every external identity's leave-one-out centroid."""
    by_identity: dict[str, list[ExternalFace]] = {}
    for f in faces:
        by_identity.setdefault(f.identity_id, []).append(f)
    records: list[ScoreRecord] = []
    for probe in faces:
        assert probe.face.quality is not None
        q = probe.face.quality
        vector = _unit(np.asarray(probe.embedding, dtype=np.float64))
        for identity in sorted(by_identity):
            centroid = _centroid(
                [f.embedding for f in by_identity[identity] if f.probe_id != probe.probe_id]
            )
            if centroid is None:
                continue
            records.append(
                ScoreRecord(
                    probe_id=probe.probe_id,
                    episode_id=f"external|{probe.identity_id}",
                    identity_id=f"external:{identity}",
                    probe_identity_id=f"external:{probe.identity_id}",
                    label=int(identity == probe.identity_id),
                    similarity=float(np.dot(vector, centroid)),
                    quality_score=q.score,
                    resolution=q.resolution,
                    sharpness=q.sharpness,
                    detection=q.detection,
                    pitch=q.pitch,
                    camera_id="external",
                    label_source="external_benchmark",
                    dataset="external",
                )
            )
    return records


# --- writing -------------------------------------------------------------------------------


def write_manifest(records: Sequence[ScoreRecord], meta: ManifestMeta, output: Path) -> None:
    """Write ``manifest.jsonl`` and ``manifest.meta.json`` (dataset hash, counts, provenance)."""
    lines = [json.dumps(asdict(r), sort_keys=True) for r in records]
    body = "\n".join(lines) + ("\n" if lines else "")
    output.write_text(body)
    meta.records = len(records)
    labels = Counter(r.label for r in records)
    meta.genuine, meta.impostor = labels[1], labels[0]
    meta.dataset_hash = hashlib.sha256(body.encode()).hexdigest()
    output.with_suffix(".meta.json").write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))


def read_manifest(path: Path) -> tuple[list[ScoreRecord], dict[str, Any]]:
    records = [
        ScoreRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]
    return records, json.loads(path.with_suffix(".meta.json").read_text())
