"""Face templates: one embedding and one quality reading from several faces of one person.

A walking person yields many small faces, each weak on its own. A template combines the best of
them into a single embedding that is matched like a face. Two rules live here so the calibration
manifest builder and the ``/match-template`` endpoint cannot diverge:

- ``TemplateAggregator``: how member embeddings combine. ``QualityWeightedMeanAggregator``
  (``qwmean-v1``) is the standard set-based baseline: the quality-weighted mean of L2-normalized
  embeddings, renormalized. A learned aggregator (for example CAFace) replaces it behind a new
  version string; a calibration artifact binds to the version it was fitted with.
- ``template_quality``: the quality features a template is calibrated on, the means over its
  ``TEMPLATE_QUALITY_TOP_K`` best members by quality score.

Embeddings travel between services as 512 little-endian float16 values in base64
(``encode_embedding_f16`` / ``decode_embedding_f16``). They are biometric data: callers must not
log or persist them.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from app.services.face_quality import FaceQuality

EMBEDDING_DIM = 512
# Template quality is the mean over this many best members: enough to smooth one lucky face,
# few enough that a template of three good faces is not diluted by its weakest ones.
TEMPLATE_QUALITY_TOP_K = 3
_F16_LE = np.dtype("<f2")


class TemplateAggregator(Protocol):
    """Combines L2-normalized member embeddings into one L2-normalized template embedding."""

    version: str

    def aggregate(
        self, embeddings: npt.NDArray[np.float32], weights: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]: ...


class QualityWeightedMeanAggregator:
    """``t = normalize(sum(w_i * normalize(e_i)))`` with each member weighted by its quality score."""

    version = "qwmean-v1"

    def aggregate(
        self, embeddings: npt.NDArray[np.float32], weights: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Raise ``ValueError`` on empty or mismatched input, non-finite values, a zero-length
        member, negative weights, all-zero weights, or members that cancel out."""
        vectors = np.asarray(embeddings, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("a template needs at least one member embedding")
        if w.shape != (vectors.shape[0],):
            raise ValueError("one weight per member embedding is required")
        if not (np.all(np.isfinite(vectors)) and np.all(np.isfinite(w))):
            raise ValueError("embeddings and weights must be finite")
        if np.any(w < 0.0):
            raise ValueError("weights must not be negative")
        if float(w.sum()) <= 0.0:
            raise ValueError("at least one member needs a positive weight")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError("a member embedding has zero length")
        template = (w[:, None] * (vectors / norms[:, None])).sum(axis=0)
        length = float(np.linalg.norm(template))
        if length <= 1e-12:
            raise ValueError("member embeddings cancel out")
        return (template / length).astype(np.float32)


@dataclass(frozen=True)
class TemplateQuality:
    """The quality features of a template: means over its best members, all in [0, 1]."""

    score: float
    resolution: float
    sharpness: float
    detection: float
    pitch: float


def template_quality(qualities: Sequence[FaceQuality]) -> TemplateQuality:
    """Means of every quality component over the ``TEMPLATE_QUALITY_TOP_K`` best members.

    Members are ranked by quality score (ties keep input order); all components come from the same
    members, so the features describe one set of faces.
    """
    if not qualities:
        raise ValueError("a template needs at least one scored member")
    best = sorted(qualities, key=lambda q: q.score, reverse=True)[:TEMPLATE_QUALITY_TOP_K]
    n = len(best)
    return TemplateQuality(
        score=sum(q.score for q in best) / n,
        resolution=sum(q.resolution for q in best) / n,
        sharpness=sum(q.sharpness for q in best) / n,
        detection=sum(q.detection for q in best) / n,
        pitch=sum(q.pitch for q in best) / n,
    )


def encode_embedding_f16(embedding: npt.NDArray[np.floating]) -> str:
    """Base64 of the embedding as little-endian float16, the wire form between services."""
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding must have {EMBEDDING_DIM} values")
    return base64.b64encode(vector.astype(_F16_LE).tobytes()).decode("ascii")


def decode_embedding_f16(encoded: str) -> npt.NDArray[np.float32]:
    """Inverse of ``encode_embedding_f16``; ``ValueError`` on bad base64, length, or values."""
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("embedding is not valid base64") from exc
    if len(raw) != EMBEDDING_DIM * _F16_LE.itemsize:
        raise ValueError(f"embedding must hold {EMBEDDING_DIM} float16 values")
    vector = np.frombuffer(raw, dtype=_F16_LE).astype(np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding values must be finite")
    return vector
