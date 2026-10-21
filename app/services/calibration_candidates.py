"""Calibration candidates: sampled face captures an operator labels to build the calibration set.

``CalibrationCandidateRepository`` is the contract; the Postgres implementation holds every query
and the in-memory implementation is kept honest by a parity test. Crops live in MinIO under
``crop_object_key``; the repository returns the keys it removed so the caller deletes the crops.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, Protocol

import asyncpg
import numpy as np

LabelState = Literal["unlabeled", "labeled", "rejected"]
LabelKind = Literal["member", "not_enrolled", "not_a_face"]
LabelSource = Literal[
    "operator_reviewed", "enrollment_capture", "external_benchmark", "sensor_attested"
]
CaptureSource = Literal["cts_identify", "enrollment", "external"]

# Sampling bucket: (quality decile 0-9, nearest enrolled identity or NO_NEAREST, camera id).
BucketKey = tuple[int, str, str]
NO_NEAREST = "none"


def quality_decile(quality_score: float) -> int:
    """The 0-9 decile of a quality score in [0, 1] (1.0 falls in decile 9)."""
    return min(int(quality_score * 10), 9)


@dataclass(frozen=True, eq=False)
class CalibrationCandidate:
    """One captured face. Similarities are raw cosine; quality is the person-id scorer's."""

    candidate_id: str
    captured_at: datetime
    camera_id: str
    source: CaptureSource
    crop_object_key: str
    embedding: np.ndarray
    arcface_model_version: str
    preprocessing_version: str
    quality_version: str
    quality_score: float
    quality_components: Mapping[str, float]
    # How the tracker cut and encoded the crop it sent (padding fractions, JPEG quality); these
    # shape the similarity distribution, so a fit never mixes profiles.
    capture_profile: Mapping[str, float | int]
    interocular_px: float | None
    yaw_deg: float
    pitch_deg: float
    nearest_person_id: str | None
    nearest_similarity: float
    label_state: LabelState = "unlabeled"
    label_identity_id: str | None = None
    label_kind: LabelKind | None = None
    label_source: LabelSource | None = None
    labeled_by: str | None = None
    labeled_at: datetime | None = None

    @property
    def bucket(self) -> BucketKey:
        return (
            quality_decile(self.quality_score),
            self.nearest_person_id or NO_NEAREST,
            self.camera_id,
        )


@dataclass(frozen=True)
class CandidatePage:
    candidates: list[CalibrationCandidate] = field(default_factory=list)
    # Opaque keyset cursor for the next page; None on the last page.
    next_cursor: str | None = None


class CalibrationCandidateRepository(Protocol):
    async def insert(self, candidate: CalibrationCandidate) -> None: ...

    async def get(self, candidate_id: str) -> CalibrationCandidate | None: ...

    async def list_page(
        self, *, state: LabelState | None, limit: int, cursor: str | None
    ) -> CandidatePage:
        """Candidates newest first (``captured_at``, then id), optionally in one state."""
        ...

    async def label(
        self,
        candidate_id: str,
        *,
        kind: LabelKind,
        identity_id: str | None,
        source: LabelSource,
        labeled_by: str,
        labeled_at: datetime,
    ) -> CalibrationCandidate | None:
        """Label (or relabel) a candidate; None when it does not exist."""
        ...

    async def reject(self, candidate_id: str) -> CalibrationCandidate | None:
        """Mark a candidate unusable and clear any label; None when it does not exist."""
        ...

    async def delete(self, candidate_id: str) -> str | None:
        """Delete a candidate; returns its crop key, or None when it did not exist."""
        ...

    async def purge(self, captured_before: datetime) -> list[str]:
        """Delete unlabeled and rejected candidates captured before the cutoff; returns crop keys."""
        ...

    async def bucket_counts(self, captured_since: datetime) -> dict[BucketKey, int]:
        """Captures per sampling bucket since a time (seeds the in-memory daily caps)."""
        ...

    async def list_labeled(self) -> list[CalibrationCandidate]:
        """Every labeled candidate, oldest first: the calibration dataset."""
        ...

    async def count_by_state(self) -> dict[LabelState, int]: ...


def _encode_cursor(candidate: CalibrationCandidate) -> str:
    return f"{candidate.captured_at.isoformat()}|{candidate.candidate_id}"


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        when, candidate_id = cursor.split("|", 1)
        return datetime.fromisoformat(when), candidate_id
    except ValueError as exc:
        raise ValueError(f"invalid cursor {cursor!r}") from exc


def _sort_key(candidate: CalibrationCandidate) -> tuple[datetime, str]:
    return (candidate.captured_at, candidate.candidate_id)


_COLUMNS = """candidate_id, captured_at, camera_id, source, crop_object_key, embedding,
    arcface_model_version, preprocessing_version, quality_version, quality_score,
    quality_components, capture_profile, interocular_px, yaw_deg, pitch_deg, nearest_person_id,
    nearest_similarity, label_state, label_identity_id, label_kind, label_source, labeled_by,
    labeled_at"""

_SQL_INSERT = f"""INSERT INTO calibration_candidates ({_COLUMNS})
VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13, $14, $15,
        $16, $17, $18, $19, $20, $21, $22, $23)"""

_SQL_SELECT = f"SELECT {_COLUMNS} FROM calibration_candidates"

_SQL_LABEL = f"""UPDATE calibration_candidates
SET label_state = 'labeled', label_kind = $2, label_identity_id = $3, label_source = $4,
    labeled_by = $5, labeled_at = $6
WHERE candidate_id = $1::uuid
RETURNING {_COLUMNS}"""

_SQL_REJECT = f"""UPDATE calibration_candidates
SET label_state = 'rejected', label_kind = NULL, label_identity_id = NULL, label_source = NULL,
    labeled_by = NULL, labeled_at = NULL
WHERE candidate_id = $1::uuid
RETURNING {_COLUMNS}"""


def _from_row(row: asyncpg.Record) -> CalibrationCandidate:
    def _json(value: Any) -> dict[str, Any]:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise ValueError("calibration candidate JSON column is not an object")
        return parsed

    return CalibrationCandidate(
        candidate_id=str(row["candidate_id"]),
        captured_at=row["captured_at"],
        camera_id=row["camera_id"],
        source=row["source"],
        crop_object_key=row["crop_object_key"],
        embedding=np.asarray(row["embedding"], dtype=np.float32),
        arcface_model_version=row["arcface_model_version"],
        preprocessing_version=row["preprocessing_version"],
        quality_version=row["quality_version"],
        quality_score=float(row["quality_score"]),
        quality_components=_json(row["quality_components"]),
        capture_profile=_json(row["capture_profile"]),
        interocular_px=None if row["interocular_px"] is None else float(row["interocular_px"]),
        yaw_deg=float(row["yaw_deg"]),
        pitch_deg=float(row["pitch_deg"]),
        nearest_person_id=row["nearest_person_id"],
        nearest_similarity=float(row["nearest_similarity"]),
        label_state=row["label_state"],
        label_identity_id=row["label_identity_id"],
        label_kind=row["label_kind"],
        label_source=row["label_source"],
        labeled_by=row["labeled_by"],
        labeled_at=row["labeled_at"],
    )


class PostgresCalibrationCandidateRepository:
    """Calibration candidates in PostgreSQL; the pool must register the pgvector codec."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert(self, candidate: CalibrationCandidate) -> None:
        c = candidate
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SQL_INSERT,
                c.candidate_id,
                c.captured_at,
                c.camera_id,
                c.source,
                c.crop_object_key,
                c.embedding,
                c.arcface_model_version,
                c.preprocessing_version,
                c.quality_version,
                c.quality_score,
                json.dumps(dict(c.quality_components)),
                json.dumps(dict(c.capture_profile)),
                c.interocular_px,
                c.yaw_deg,
                c.pitch_deg,
                c.nearest_person_id,
                c.nearest_similarity,
                c.label_state,
                c.label_identity_id,
                c.label_kind,
                c.label_source,
                c.labeled_by,
                c.labeled_at,
            )

    async def get(self, candidate_id: str) -> CalibrationCandidate | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"{_SQL_SELECT} WHERE candidate_id = $1::uuid", candidate_id)
        return _from_row(row) if row else None

    async def list_page(
        self, *, state: LabelState | None, limit: int, cursor: str | None
    ) -> CandidatePage:
        after_time, after_id = _decode_cursor(cursor) if cursor else (None, None)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""{_SQL_SELECT}
                WHERE ($1::text IS NULL OR label_state = $1)
                  AND ($2::timestamptz IS NULL OR (captured_at, candidate_id) < ($2, $3::uuid))
                ORDER BY captured_at DESC, candidate_id DESC
                LIMIT $4""",
                state,
                after_time,
                after_id,
                limit + 1,
            )
        candidates = [_from_row(r) for r in rows[:limit]]
        more = len(rows) > limit
        return CandidatePage(candidates, _encode_cursor(candidates[-1]) if more else None)

    async def label(
        self,
        candidate_id: str,
        *,
        kind: LabelKind,
        identity_id: str | None,
        source: LabelSource,
        labeled_by: str,
        labeled_at: datetime,
    ) -> CalibrationCandidate | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _SQL_LABEL, candidate_id, kind, identity_id, source, labeled_by, labeled_at
            )
        return _from_row(row) if row else None

    async def reject(self, candidate_id: str) -> CalibrationCandidate | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_REJECT, candidate_id)
        return _from_row(row) if row else None

    async def delete(self, candidate_id: str) -> str | None:
        async with self._pool.acquire() as conn:
            key = await conn.fetchval(
                "DELETE FROM calibration_candidates WHERE candidate_id = $1::uuid "
                "RETURNING crop_object_key",
                candidate_id,
            )
        return None if key is None else str(key)

    async def purge(self, captured_before: datetime) -> list[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """DELETE FROM calibration_candidates
                   WHERE label_state IN ('unlabeled', 'rejected') AND captured_at < $1
                   RETURNING crop_object_key""",
                captured_before,
            )
        return sorted(str(r["crop_object_key"]) for r in rows)

    async def bucket_counts(self, captured_since: datetime) -> dict[BucketKey, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT LEAST(floor(quality_score * 10)::int, 9) AS decile,
                          COALESCE(nearest_person_id, $2) AS nearest,
                          camera_id, count(*)::int AS n
                   FROM calibration_candidates
                   WHERE captured_at >= $1
                   GROUP BY 1, 2, 3""",
                captured_since,
                NO_NEAREST,
            )
        return {(r["decile"], r["nearest"], r["camera_id"]): r["n"] for r in rows}

    async def list_labeled(self) -> list[CalibrationCandidate]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"{_SQL_SELECT} WHERE label_state = 'labeled' ORDER BY captured_at, candidate_id"
            )
        return [_from_row(r) for r in rows]

    async def count_by_state(self) -> dict[LabelState, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT label_state, count(*)::int AS n FROM calibration_candidates GROUP BY 1"
            )
        counts: dict[LabelState, int] = {"unlabeled": 0, "labeled": 0, "rejected": 0}
        counts.update({r["label_state"]: r["n"] for r in rows})
        return counts


class InMemoryCalibrationCandidateRepository:
    """Dict-backed peer for unit tests; parity-tested against Postgres."""

    def __init__(self) -> None:
        self._rows: dict[str, CalibrationCandidate] = {}

    async def insert(self, candidate: CalibrationCandidate) -> None:
        if candidate.candidate_id in self._rows:
            raise ValueError(f"duplicate candidate {candidate.candidate_id}")
        self._rows[candidate.candidate_id] = candidate

    async def get(self, candidate_id: str) -> CalibrationCandidate | None:
        return self._rows.get(candidate_id)

    async def list_page(
        self, *, state: LabelState | None, limit: int, cursor: str | None
    ) -> CandidatePage:
        rows = sorted(
            (c for c in self._rows.values() if state is None or c.label_state == state),
            key=_sort_key,
            reverse=True,
        )
        if cursor:
            after = _decode_cursor(cursor)
            rows = [c for c in rows if _sort_key(c) < after]
        page = rows[:limit]
        more = len(rows) > limit
        return CandidatePage(page, _encode_cursor(page[-1]) if more else None)

    async def label(
        self,
        candidate_id: str,
        *,
        kind: LabelKind,
        identity_id: str | None,
        source: LabelSource,
        labeled_by: str,
        labeled_at: datetime,
    ) -> CalibrationCandidate | None:
        current = self._rows.get(candidate_id)
        if current is None:
            return None
        updated = replace(
            current,
            label_state="labeled",
            label_kind=kind,
            label_identity_id=identity_id,
            label_source=source,
            labeled_by=labeled_by,
            labeled_at=labeled_at,
        )
        self._rows[candidate_id] = updated
        return updated

    async def reject(self, candidate_id: str) -> CalibrationCandidate | None:
        current = self._rows.get(candidate_id)
        if current is None:
            return None
        updated = replace(
            current,
            label_state="rejected",
            label_kind=None,
            label_identity_id=None,
            label_source=None,
            labeled_by=None,
            labeled_at=None,
        )
        self._rows[candidate_id] = updated
        return updated

    async def delete(self, candidate_id: str) -> str | None:
        removed = self._rows.pop(candidate_id, None)
        return None if removed is None else removed.crop_object_key

    async def purge(self, captured_before: datetime) -> list[str]:
        expired = [
            c
            for c in self._rows.values()
            if c.label_state in ("unlabeled", "rejected") and c.captured_at < captured_before
        ]
        for c in expired:
            del self._rows[c.candidate_id]
        return sorted(c.crop_object_key for c in expired)

    async def bucket_counts(self, captured_since: datetime) -> dict[BucketKey, int]:
        counts: dict[BucketKey, int] = {}
        for c in self._rows.values():
            if c.captured_at >= captured_since:
                counts[c.bucket] = counts.get(c.bucket, 0) + 1
        return counts

    async def list_labeled(self) -> list[CalibrationCandidate]:
        return sorted((c for c in self._rows.values() if c.label_state == "labeled"), key=_sort_key)

    async def count_by_state(self) -> dict[LabelState, int]:
        counts: dict[LabelState, int] = {"unlabeled": 0, "labeled": 0, "rejected": 0}
        for c in self._rows.values():
            counts[c.label_state] += 1
        return counts
