"""Suggest face quality ramp anchors from operator-labeled face images.

Usage:
    uv run --extra calibration-tools python scripts/calibrate_face_quality.py MANIFEST.json

The manifest lists image files and the enrolled identity each shows:

    {"schema_version": 1, "faces": [{"image": "alpha/001.jpg", "identity_id": "resident-alpha"}]}

Image paths are relative to the manifest. Manifests and images are private household data: keep
them outside the repository. Detection runs through ``FaceEngine`` directly (never ``/identify``,
which records visitor sightings), and enrolled centroids are read with a read-only query. The
largest face in each image is measured. The script prints a report and suggested anchors; a human
decides whether to change ``face_quality.heuristic`` in ``config/settings.yaml``.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import asyncpg
import cv2
import numpy as np
from pgvector.asyncpg import register_vector
from triton_shared.client import TritonGrpcClient

from app import config
from app.calibration.quality_breakpoints import FaceQualitySample, InputReport, analyze
from app.services.face_engine import FaceEngine, align_face
from app.services.face_quality import build_face_quality_scorer, laplacian_variance

_SCHEMA_VERSION = 1


def _load_manifest(path: Path) -> list[tuple[Path, str]]:
    data = json.loads(path.read_text())
    if data.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"manifest schema_version must be {_SCHEMA_VERSION}")
    entries = [(path.parent / f["image"], str(f["identity_id"])) for f in data["faces"]]
    missing = [str(p) for p, _ in entries if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest images not found: {missing[:5]}")
    return entries


async def _centroids(dsn: str, identity_ids: set[str]) -> dict[str, np.ndarray]:
    conn = await asyncpg.connect(dsn)
    try:
        await register_vector(conn)
        # Read-only: this tool never writes enrollment or any other table.
        async with conn.transaction(readonly=True):
            rows = await conn.fetch(
                "SELECT person_id, centroid FROM centroids WHERE person_id = ANY($1::text[])",
                sorted(identity_ids),
            )
    finally:
        await conn.close()
    return {row["person_id"]: np.asarray(row["centroid"], dtype=np.float32) for row in rows}


async def _samples(entries: list[tuple[Path, str]]) -> tuple[list[FaceQualitySample], int]:
    dsn = config.get("database.dsn")
    if not dsn:
        raise RuntimeError("database.dsn is not configured (set DATABASE_URL)")
    centroids = await _centroids(dsn, {identity for _, identity in entries})
    unknown = sorted({identity for _, identity in entries} - set(centroids))
    if unknown:
        raise ValueError(f"identities not enrolled: {unknown}")

    scorer = build_face_quality_scorer(config.get("face_quality", {}))
    client = TritonGrpcClient(
        str(config.get("face_engine.triton_url", "triton:8701")),
        timeout_ms=int(config.get("face_engine.triton_timeout_ms", 30_000)),
    )
    samples: list[FaceQualitySample] = []
    skipped = 0
    async with client:
        engine = FaceEngine(client, scorer)
        await engine.validate_models()
        for image_path, identity in entries:
            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue
            faces = await engine.detect_faces(image)
            if not faces:
                skipped += 1
                continue
            face = max(faces, key=lambda f: f.geometry.face_width_px * f.geometry.face_height_px)
            assert face.keypoints_5_px is not None  # set by FaceEngine for every face
            gray = np.asarray(
                cv2.cvtColor(align_face(image, face.keypoints_5_px), cv2.COLOR_BGR2GRAY),
                dtype=np.uint8,
            )
            samples.append(
                FaceQualitySample(
                    interocular_px=face.geometry.interocular_px,
                    det_score=face.det_score,
                    abs_pitch_deg=abs(face.pitch_deg),
                    log10_laplacian_variance=math.log10(max(laplacian_variance(gray), 1e-6)),
                    raw_similarity=FaceEngine.compute_similarity(
                        face.embedding, centroids[identity]
                    ),
                )
            )
    return samples, skipped


def _print(reports: list[InputReport], measured: int, skipped: int) -> None:
    print(f"Measured {measured} labeled faces ({skipped} images unreadable or without a face).")
    print("input                      n     p5      p50     p95     rho     p-value  low     high")
    for r in reports:
        rho = "n/a" if r.spearman_rho is None else f"{r.spearman_rho:+.3f}"
        p_value = "n/a" if r.spearman_p_value is None else f"{r.spearman_p_value:.1e}"
        print(
            f"{r.name:<26} {r.count:<5} {r.p5:<7.3f} {r.p50:<7.3f} {r.p95:<7.3f} "
            f"{rho:<7} {p_value:<8} {r.suggested_low:<7.3f} {r.suggested_high:.3f}"
        )
    print(
        "A component whose rho is near zero does not predict recognizability on these cameras; "
        "review its weight before changing anchors. Changing settings means a new scorer version."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    samples, skipped = asyncio.run(_samples(_load_manifest(args.manifest)))
    _print(analyze(samples), len(samples), skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
