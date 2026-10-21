"""Manifest v2: episodes, leave-episode-out galleries, open-set impostors, provenance checks."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.calibration.episodes import EpisodeInput, assign_episodes
from app.calibration.manifest import (
    ExternalFace,
    GalleryEmbedding,
    ManifestError,
    ManifestMeta,
    detect_external_faces,
    external_records,
    household_records,
    load_external_listing,
    read_manifest,
    write_manifest,
)
from app.services.calibration_candidates import CalibrationCandidate
from app.services.face_models import DetectedFace
from app.services.face_quality import FaceQuality

_T = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _candidate(
    index: int, *, kind: str | None, identity: str | None = None, minutes: float = 0, **changes: Any
) -> CalibrationCandidate:
    fields: dict[str, Any] = {
        "candidate_id": str(uuid.UUID(int=index + 1)),
        "captured_at": _T + timedelta(minutes=minutes),
        "camera_id": "cam-1",
        "source": "cts_identify",
        "crop_object_key": f"calibration/{index}.png",
        "embedding": _unit(index),
        "arcface_model_version": "buffalo_l",
        "preprocessing_version": "v1",
        "quality_version": "heuristic-v1",
        "quality_score": 0.4,
        "quality_components": {"resolution": 0.3, "sharpness": 0.5, "detection": 0.8, "pitch": 1.0},
        "capture_profile": {
            "crop_pad_top_frac": 0.15,
            "crop_pad_side_frac": 0.1,
            "crop_jpeg_quality": 95,
        },
        "interocular_px": 20.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "nearest_person_id": None,
        "nearest_similarity": 0.3,
        "label_state": "labeled" if kind else "unlabeled",
        "label_kind": kind,
        "label_identity_id": identity,
        "label_source": "operator_reviewed" if kind else None,
        "labeled_by": "op" if kind else None,
        "labeled_at": _T if kind else None,
    }
    fields.update(changes)
    return CalibrationCandidate(**fields)


_GALLERY = {
    "alpha": [GalleryEmbedding(_unit(100)), GalleryEmbedding(_unit(101))],
    "beta": [GalleryEmbedding(_unit(200))],
}


def test_episodes_split_on_gaps_and_per_camera_and_label() -> None:
    def inp(i: int, minutes: float, camera: str = "cam-1", label: str = "alpha") -> EpisodeInput:
        return EpisodeInput(str(i), _T + timedelta(minutes=minutes), camera, label)

    episodes = assign_episodes(
        [
            inp(1, 0),
            inp(2, 4.9),
            inp(3, 9.8),
            inp(4, 15),
            inp(5, 1, camera="cam-2"),
            inp(6, 1, label="beta"),
        ],
        gap_s=300,
    )
    assert episodes["1"] == episodes["2"] == episodes["3"]  # gaps under 5 minutes chain
    assert episodes["4"] != episodes["1"]  # a 5.2 minute gap starts a new walk
    assert len({episodes["1"], episodes["5"], episodes["6"]}) == 3


def test_member_probe_scores_every_identity_with_one_genuine() -> None:
    meta = ManifestMeta()
    records = household_records(
        [_candidate(1, kind="member", identity="alpha")], _GALLERY, episode_gap_s=300, meta=meta
    )
    assert [(r.identity_id, r.label) for r in records] == [("alpha", 1), ("beta", 0)]
    assert records[0].probe_identity_id == "alpha"
    assert meta.probes_by_identity == {"alpha": 1}
    centroid = np.mean([_unit(100), _unit(101)], axis=0)
    centroid /= np.linalg.norm(centroid)
    assert records[0].similarity == pytest.approx(float(np.dot(_unit(1), centroid)), abs=1e-6)


def test_open_set_probes_produce_only_impostor_records() -> None:
    meta = ManifestMeta()
    records = household_records(
        [_candidate(2, kind="not_enrolled")], _GALLERY, episode_gap_s=300, meta=meta
    )
    assert {r.label for r in records} == {0}
    assert {r.probe_identity_id for r in records} == {None}
    assert len(records) == 2
    assert meta.open_set_probes == 1


def test_leave_episode_out_drops_gallery_embeddings_from_the_probe_episode() -> None:
    probe = _candidate(1, kind="member", identity="alpha")
    walk_mate = _candidate(2, kind="member", identity="alpha", minutes=1)
    gallery = {
        "alpha": [
            GalleryEmbedding(_unit(100)),
            GalleryEmbedding(walk_mate.embedding, walk_mate.candidate_id),
        ],
        "beta": [GalleryEmbedding(_unit(200))],
    }
    records = household_records([probe, walk_mate], gallery, episode_gap_s=300, meta=ManifestMeta())
    genuine = next(r for r in records if r.probe_id == probe.candidate_id and r.label == 1)
    # Only the photo embedding remains: the walk-mate's promoted embedding shares the episode.
    assert genuine.similarity == pytest.approx(float(np.dot(_unit(1), _unit(100))), abs=1e-6)


def test_not_a_face_and_unenrolled_identities_are_skipped() -> None:
    meta = ManifestMeta()
    records = household_records(
        [
            _candidate(1, kind="not_a_face"),
            _candidate(2, kind="member", identity="gone"),
            _candidate(3, kind=None),
        ],
        _GALLERY,
        episode_gap_s=300,
        meta=meta,
    )
    assert records == []
    assert (meta.skipped_not_a_face, meta.skipped_identity_not_enrolled) == (1, 1)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "capture_profile": {
                "crop_pad_top_frac": 0.0,
                "crop_pad_side_frac": 0.0,
                "crop_jpeg_quality": 90,
            }
        },
        {"quality_version": "heuristic-v2"},
        {"arcface_model_version": "other"},
    ],
)
def test_mixed_provenance_is_refused(changes: dict[str, Any]) -> None:
    candidates = [
        _candidate(1, kind="member", identity="alpha"),
        _candidate(2, kind="not_enrolled", **changes),
    ]
    with pytest.raises(ManifestError, match="mix"):
        household_records(candidates, _GALLERY, episode_gap_s=300, meta=ManifestMeta())


def test_external_listing_requires_a_complete_licence(tmp_path: Path) -> None:
    listing = tmp_path / "set.jsonl"
    listing.write_text(json.dumps({"image": "a/1.jpg", "identity_id": "x"}) + "\n")
    with pytest.raises(ManifestError, match="licence sidecar"):
        load_external_listing(listing)
    sidecar = tmp_path / "set.jsonl.licence.json"
    sidecar.write_text(json.dumps({"name": "Example", "version": "1"}))
    with pytest.raises(ManifestError, match="licence"):
        load_external_listing(listing)
    sidecar.write_text(json.dumps({"name": "Example", "version": "1", "licence": "research only"}))
    entries, info = load_external_listing(listing)
    assert entries == [(tmp_path / "a" / "1.jpg", "x")]
    assert info == {"name": "Example", "version": "1", "licence": "research only"}


def _scored_face(seed: int) -> DetectedFace:
    face = DetectedFace(bbox=[0.0, 0.0, 40.0, 40.0], embedding=_unit(seed), det_score=0.9)
    face.quality = FaceQuality(
        score=0.3, resolution=0.2, sharpness=0.4, detection=0.9, pitch=1.0, version="heuristic-v1"
    )
    return face


def test_external_records_use_leave_one_out_centroids() -> None:
    faces = [
        ExternalFace(f"e{i}", identity, _unit(i), _scored_face(i))
        for i, identity in enumerate(["p", "p", "q"])
    ]
    records = external_records(faces)
    # q has one image: it has no leave-one-out centroid for itself, so no genuine record.
    assert sorted((r.probe_id, r.identity_id, r.label) for r in records) == [
        ("e0", "external:p", 1),
        ("e0", "external:q", 0),
        ("e1", "external:p", 1),
        ("e1", "external:q", 0),
        ("e2", "external:p", 0),
    ]
    assert {r.dataset for r in records} == {"external"}
    first = next(r for r in records if r.probe_id == "e0" and r.label == 1)
    assert first.similarity == pytest.approx(float(np.dot(_unit(0), _unit(1))), abs=1e-6)


async def test_external_images_without_a_face_are_counted(tmp_path: Path) -> None:
    import cv2

    image = tmp_path / "a.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))

    class _Detector:
        def __init__(self) -> None:
            self.calls = 0

        async def detect_faces(self, img: np.ndarray) -> list[DetectedFace]:
            self.calls += 1
            return [_scored_face(1)] if self.calls == 1 else []

    faces, dropped = await detect_external_faces(
        [(image, "p"), (image, "p"), (tmp_path / "missing.png", "p")], _Detector()
    )
    assert (len(faces), dropped) == (1, 2)


def test_manifest_round_trips_with_a_content_hash(tmp_path: Path) -> None:
    meta = ManifestMeta()
    records = household_records(
        [_candidate(1, kind="member", identity="alpha")], _GALLERY, episode_gap_s=300, meta=meta
    )
    output = tmp_path / "manifest.jsonl"
    write_manifest(records, meta, output)
    loaded, loaded_meta = read_manifest(output)
    assert loaded == records
    assert (loaded_meta["records"], loaded_meta["genuine"], loaded_meta["impostor"]) == (2, 1, 1)
    assert loaded_meta["capture_profile"] == {
        "crop_pad_top_frac": 0.15,
        "crop_pad_side_frac": 0.1,
        "crop_jpeg_quality": 95,
    }
    assert len(loaded_meta["dataset_hash"]) == 64
