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

The ``template`` profile calibrates aggregated templates instead of single faces: each record is a
template of k faces (k from 2 to 8) drawn from one episode, aggregated and quality-scored by the
same functions ``/match-template`` uses, and scored like a probe. Templates of one episode are
highly correlated, so at most ``max_per_episode`` are drawn from each. Numpy only; offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import cv2
import numpy as np

from app.calibration.episodes import EpisodeInput, assign_episodes, episode_label
from app.calibration.models import GalleryRepresentation
from app.services.calibration_candidates import CalibrationCandidate
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality
from app.services.prototype_builder import (
    EmbeddingSource,
    MemberEmbedding,
    PoseBin,
    PrototypePolicy,
    build_prototypes,
)
from app.services.template_aggregation import TemplateAggregator, template_quality

Dataset = Literal["household", "external"]
Profile = Literal["single_frame", "template"]
QUALITY_FIELDS = ("quality_score", "resolution", "sharpness", "detection", "pitch")
# Template records: member counts drawn per template, and the cap on templates per episode.
TEMPLATE_MIN_MEMBERS = 2
TEMPLATE_MAX_MEMBERS = 8
DEFAULT_TEMPLATES_PER_EPISODE = 20


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
    # Top-1 margin relative to the strongest other enrolled identity; None in pre-margin manifests.
    margin: float | None = None


@dataclass(frozen=True)
class GalleryEmbedding:
    embedding: np.ndarray
    # The calibration candidate the embedding was promoted from, when it was; None for photos.
    source_candidate_id: str | None = None
    pose_bin: PoseBin | None = None
    source: EmbeddingSource = "enrollment_photo"
    quality_score: float | None = None
    quality_version: str | None = None


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
    profile: Profile = "single_frame"
    # Template profile only: the aggregator version and the per-episode template cap.
    aggregation: str | None = None
    max_records_per_episode: int | None = None
    # Gallery representation whose score distribution this manifest records.
    gallery_representation: GalleryRepresentation = "centroid"
    gallery_revisions: dict[str, int] = field(default_factory=dict)


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


def _prototype_vectors(
    identity: str,
    embeddings: Sequence[GalleryEmbedding],
    policy: PrototypePolicy,
) -> list[np.ndarray]:
    if not embeddings:
        return []
    members = [
        MemberEmbedding(
            embedding=g.embedding,
            embedding_id=idx,
            pose_bin=g.pose_bin,
            source=g.source,
            quality_score=g.quality_score,
            quality_version=g.quality_version,
            source_candidate_id=g.source_candidate_id,
        )
        for idx, g in enumerate(embeddings)
    ]
    return [_unit(p.embedding) for p in build_prototypes(identity, members, policy)]


def _scores_with_margins(scores: Sequence[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """Attach each scored identity's margin vs the highest-scoring other identity."""
    if len(scores) <= 1:
        return [(identity, sim, sim) for identity, sim in scores]
    result: list[tuple[str, float, float]] = []
    for identity, sim in scores:
        other_best = max(other_sim for other_id, other_sim in scores if other_id != identity)
        result.append((identity, sim, sim - other_best))
    return result


@dataclass
class _HouseholdProbes:
    usable: list[CalibrationCandidate]
    episodes: dict[str, str]  # candidate id -> episode id
    episode_members: dict[str, set[str]]  # episode id -> candidate ids
    full_centroids: dict[str, np.ndarray | None]
    full_prototypes: dict[str, list[np.ndarray]]


def _household_probes(
    candidates: Sequence[CalibrationCandidate],
    gallery: Mapping[str, Sequence[GalleryEmbedding]],
    *,
    episode_gap_s: float,
    meta: ManifestMeta,
    gallery_representation: GalleryRepresentation = "centroid",
    prototype_policy: PrototypePolicy = PrototypePolicy(),
) -> _HouseholdProbes | None:
    """Usable labeled captures, their episodes, and precomputed gallery representations."""
    meta.gallery_representation = gallery_representation
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
        return None

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
    full_prototypes = (
        {
            identity: _prototype_vectors(identity, embeddings, prototype_policy)
            for identity, embeddings in gallery.items()
        }
        if gallery_representation == "prototypes-v1"
        else {}
    )
    return _HouseholdProbes(usable, episodes, episode_members, full_centroids, full_prototypes)


def _gallery_scores(
    probe: np.ndarray,
    probe_identity: str | None,
    episode: str,
    gallery: Mapping[str, Sequence[GalleryEmbedding]],
    prepared: _HouseholdProbes,
    *,
    gallery_representation: GalleryRepresentation = "centroid",
    prototype_policy: PrototypePolicy = PrototypePolicy(),
) -> list[tuple[str, float, float]]:
    """(identity, cosine, margin) of a unit probe against every identity's gallery.

    The probe's own identity is scored against a gallery without embeddings promoted from the
    probe's episode (leave-episode-out).
    """
    scores: list[tuple[str, float]] = []
    for identity in sorted(gallery):
        if gallery_representation == "prototypes-v1":
            if identity == probe_identity:
                same_episode = prepared.episode_members[episode]
                kept = [g for g in gallery[identity] if g.source_candidate_id not in same_episode]
                vectors = _prototype_vectors(identity, kept, prototype_policy)
            else:
                vectors = prepared.full_prototypes.get(identity, [])
            if vectors:
                scores.append((identity, max(float(np.dot(probe, v)) for v in vectors)))
        else:
            centroid = prepared.full_centroids[identity]
            if identity == probe_identity:
                same_episode = prepared.episode_members[episode]
                centroid = _centroid(
                    [
                        g.embedding
                        for g in gallery[identity]
                        if g.source_candidate_id not in same_episode
                    ]
                )
            if centroid is not None:
                scores.append((identity, float(np.dot(probe, centroid))))
    return _scores_with_margins(scores)


def _probe_identity(c: CalibrationCandidate) -> str | None:
    return c.label_identity_id if c.label_kind == "member" else None


def household_records(
    candidates: Sequence[CalibrationCandidate],
    gallery: Mapping[str, Sequence[GalleryEmbedding]],
    *,
    episode_gap_s: float,
    meta: ManifestMeta,
    gallery_representation: GalleryRepresentation = "centroid",
    prototype_policy: PrototypePolicy = PrototypePolicy(),
) -> list[ScoreRecord]:
    """Score records for labeled household captures against every enrolled identity."""
    prepared = _household_probes(
        candidates,
        gallery,
        episode_gap_s=episode_gap_s,
        meta=meta,
        gallery_representation=gallery_representation,
        prototype_policy=prototype_policy,
    )
    if prepared is None:
        return []

    records: list[ScoreRecord] = []
    for c in prepared.usable:
        probe_identity = _probe_identity(c)
        episode = prepared.episodes[c.candidate_id]
        probe = _unit(np.asarray(c.embedding, dtype=np.float64))
        for identity, similarity, margin in _gallery_scores(
            probe,
            probe_identity,
            episode,
            gallery,
            prepared,
            gallery_representation=gallery_representation,
            prototype_policy=prototype_policy,
        ):
            records.append(
                ScoreRecord(
                    probe_id=c.candidate_id,
                    episode_id=episode,
                    identity_id=identity,
                    probe_identity_id=probe_identity,
                    label=int(identity == probe_identity),
                    similarity=similarity,
                    quality_score=c.quality_score,
                    resolution=float(c.quality_components["resolution"]),
                    sharpness=float(c.quality_components["sharpness"]),
                    detection=float(c.quality_components["detection"]),
                    pitch=float(c.quality_components["pitch"]),
                    camera_id=c.camera_id,
                    label_source=str(c.label_source),
                    dataset="household",
                    margin=margin,
                )
            )
    meta.open_set_probes += sum(1 for c in prepared.usable if c.label_kind == "not_enrolled")
    for c in prepared.usable:
        if c.label_identity_id is not None:
            meta.probes_by_identity[c.label_identity_id] = (
                meta.probes_by_identity.get(c.label_identity_id, 0) + 1
            )
    return records


def _candidate_quality(c: CalibrationCandidate) -> FaceQuality:
    return FaceQuality(
        score=c.quality_score,
        resolution=float(c.quality_components["resolution"]),
        sharpness=float(c.quality_components["sharpness"]),
        detection=float(c.quality_components["detection"]),
        pitch=float(c.quality_components["pitch"]),
        version=c.quality_version,
    )


def episode_subsets(
    member_count: int,
    rng: random.Random,
    *,
    max_subsets: int,
    min_members: int = TEMPLATE_MIN_MEMBERS,
    max_members: int = TEMPLATE_MAX_MEMBERS,
) -> list[tuple[int, ...]]:
    """Distinct member-index subsets of one episode, sizes drawn uniformly, at most ``max_subsets``.

    Deterministic for a seeded ``rng``. An episode with fewer than ``min_members`` faces yields
    nothing; a small episode yields every subset it has when that is fewer than ``max_subsets``.
    """
    largest = min(max_members, member_count)
    if largest < min_members:
        return []
    available = sum(math.comb(member_count, k) for k in range(min_members, largest + 1))
    wanted = min(max_subsets, available)
    chosen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    attempts = 0
    while len(ordered) < wanted and attempts < wanted * 50:
        attempts += 1
        k = rng.randint(min_members, largest)
        subset = tuple(sorted(rng.sample(range(member_count), k)))
        if subset not in chosen:
            chosen.add(subset)
            ordered.append(subset)
    return ordered


def template_records(
    candidates: Sequence[CalibrationCandidate],
    gallery: Mapping[str, Sequence[GalleryEmbedding]],
    *,
    episode_gap_s: float,
    meta: ManifestMeta,
    aggregator: TemplateAggregator,
    max_per_episode: int = DEFAULT_TEMPLATES_PER_EPISODE,
    random_seed: int = 42,
    gallery_representation: GalleryRepresentation = "centroid",
    prototype_policy: PrototypePolicy = PrototypePolicy(),
) -> list[ScoreRecord]:
    """Score records for templates drawn within labeled household episodes.

    Every template's members come from one episode (one person, one camera, one walk), so a
    template never mixes people; it is aggregated with ``aggregator`` (weights: member quality
    score) and its features come from ``template_quality``, exactly as at runtime.
    """
    meta.profile = "template"
    meta.aggregation = aggregator.version
    meta.max_records_per_episode = max_per_episode
    prepared = _household_probes(
        candidates,
        gallery,
        episode_gap_s=episode_gap_s,
        meta=meta,
        gallery_representation=gallery_representation,
        prototype_policy=prototype_policy,
    )
    if prepared is None:
        return []

    by_episode: dict[str, list[CalibrationCandidate]] = {}
    for c in prepared.usable:
        by_episode.setdefault(prepared.episodes[c.candidate_id], []).append(c)
    rng = random.Random(random_seed)
    records: list[ScoreRecord] = []
    for episode in sorted(by_episode):
        members = sorted(by_episode[episode], key=lambda c: c.candidate_id)
        probe_identity = _probe_identity(members[0])
        subsets = episode_subsets(len(members), rng, max_subsets=max_per_episode)
        for index, subset in enumerate(subsets):
            chosen = [members[i] for i in subset]
            qualities = [_candidate_quality(c) for c in chosen]
            template = aggregator.aggregate(
                np.stack([np.asarray(c.embedding, dtype=np.float32) for c in chosen]),
                np.asarray([q.score for q in qualities], dtype=np.float32),
            )
            quality = template_quality(qualities)
            probe = _unit(np.asarray(template, dtype=np.float64))
            for identity, similarity, margin in _gallery_scores(
                probe,
                probe_identity,
                episode,
                gallery,
                prepared,
                gallery_representation=gallery_representation,
                prototype_policy=prototype_policy,
            ):
                records.append(
                    ScoreRecord(
                        probe_id=f"template:{episode}:{index}",
                        episode_id=episode,
                        identity_id=identity,
                        probe_identity_id=probe_identity,
                        label=int(identity == probe_identity),
                        similarity=similarity,
                        quality_score=quality.score,
                        resolution=quality.resolution,
                        sharpness=quality.sharpness,
                        detection=quality.detection,
                        pitch=quality.pitch,
                        camera_id=chosen[0].camera_id,
                        label_source=str(chosen[0].label_source),
                        dataset="household",
                        margin=margin,
                    )
                )
            if probe_identity is None:
                meta.open_set_probes += 1
            else:
                meta.probes_by_identity[probe_identity] = (
                    meta.probes_by_identity.get(probe_identity, 0) + 1
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
        raw_scores: list[tuple[str, float]] = []
        for identity in sorted(by_identity):
            centroid = _centroid(
                [f.embedding for f in by_identity[identity] if f.probe_id != probe.probe_id]
            )
            if centroid is not None:
                raw_scores.append((identity, float(np.dot(vector, centroid))))
        for identity, similarity, margin in _scores_with_margins(raw_scores):
            records.append(
                ScoreRecord(
                    probe_id=probe.probe_id,
                    episode_id=f"external|{probe.identity_id}",
                    identity_id=f"external:{identity}",
                    probe_identity_id=f"external:{probe.identity_id}",
                    label=int(identity == probe.identity_id),
                    similarity=similarity,
                    quality_score=q.score,
                    resolution=q.resolution,
                    sharpness=q.sharpness,
                    detection=q.detection,
                    pitch=q.pitch,
                    camera_id="external",
                    label_source="external_benchmark",
                    dataset="external",
                    margin=margin,
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
