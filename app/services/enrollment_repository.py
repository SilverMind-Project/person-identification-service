"""Persistence for the enrollment gallery: members, their embeddings, and their centroids.

``EnrollmentRepository`` is the contract the enrollment service depends on. The Postgres
implementation holds every gallery query except the centroid recompute, which lives in
``app.services.centroid`` because visitor-cluster naming runs it inside its own transaction; the
repository delegates to it so there is exactly one copy of that SQL. The in-memory
implementation exists for fast unit tests and is kept honest by a parity test against Postgres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

import asyncpg
import numpy as np

from app.models.enrollment import MemberInfo
from app.services.centroid import insert_embeddings, recompute_member_centroid

UpsertStatus = Literal["enrolled", "updated"]


@dataclass(frozen=True)
class CentroidHit:
    """One enrolled member's centroid and its cosine similarity to a query embedding."""

    person_id: str
    name: str
    # Cosine similarity, 1 - pgvector cosine distance, in [-1, 1].
    similarity: float


class EnrollmentRepository(Protocol):
    async def nearest_centroids(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        """The ``k`` member centroids closest to ``embedding``, most similar first."""
        ...

    async def member_exists(self, person_id: str) -> bool: ...

    async def embedding_count(self, person_id: str) -> int: ...

    async def upsert_member(self, person_id: str, name: str) -> UpsertStatus:
        """Create the member, or rename an existing one; report which happened."""
        ...

    async def insert_embeddings(self, person_id: str, embeddings: Sequence[np.ndarray]) -> None: ...

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        """Set the member's centroid to the L2-normalized mean of all its embeddings."""
        ...

    async def list_members(self) -> list[MemberInfo]:
        """Every member with its embedding count, ordered by name."""
        ...

    async def get_member(self, person_id: str) -> MemberInfo | None: ...

    async def remove_member(self, person_id: str) -> bool:
        """Delete the member with its embeddings and centroid; False when it did not exist."""
        ...

    async def member_count(self) -> int: ...

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Every member's enrollment embeddings, oldest first (for offline calibration)."""
        ...


_SQL_NEAREST = """SELECT m.person_id, m.name,
       1.0 - (c.centroid <=> $1::vector) AS similarity
FROM centroids c
JOIN members m ON m.person_id = c.person_id
ORDER BY c.centroid <=> $1::vector
LIMIT $2"""

_SQL_MEMBERS = """SELECT m.person_id, m.name, m.created_at,
       count(e.id)::int AS embedding_count
FROM members m
LEFT JOIN embeddings e ON e.person_id = m.person_id
GROUP BY m.person_id, m.name, m.created_at
ORDER BY m.name"""

_SQL_MEMBER = """SELECT m.person_id, m.name, m.created_at,
       count(e.id)::int AS embedding_count
FROM members m
LEFT JOIN embeddings e ON e.person_id = m.person_id
WHERE m.person_id = $1
GROUP BY m.person_id, m.name, m.created_at"""


def _member_info(row: asyncpg.Record) -> MemberInfo:
    return MemberInfo(
        person_id=row["person_id"],
        name=row["name"],
        embedding_count=row["embedding_count"],
        created_at=row["created_at"],
    )


class PostgresEnrollmentRepository:
    """The gallery in PostgreSQL with pgvector; the pool must register the vector codec."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def nearest_centroids(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_NEAREST, embedding, k)
        return [
            CentroidHit(
                person_id=str(r["person_id"]),
                name=str(r["name"]),
                similarity=float(r["similarity"]),
            )
            for r in rows
        ]

    async def member_exists(self, person_id: str) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT person_id FROM members WHERE person_id = $1", person_id
            )
        return row is not None

    async def embedding_count(self, person_id: str) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT count(*) AS cnt FROM embeddings WHERE person_id = $1", person_id
            )
        return int(row["cnt"])

    async def upsert_member(self, person_id: str, name: str) -> UpsertStatus:
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT person_id FROM members WHERE person_id = $1", person_id
            )
            if existing:
                await conn.execute(
                    "UPDATE members SET name = $1 WHERE person_id = $2", name, person_id
                )
                return "updated"
            await conn.execute(
                "INSERT INTO members (person_id, name) VALUES ($1, $2)", person_id, name
            )
            return "enrolled"

    async def insert_embeddings(self, person_id: str, embeddings: Sequence[np.ndarray]) -> None:
        async with self._pool.acquire() as conn:
            await insert_embeddings(conn, person_id, list(embeddings))

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        async with self._pool.acquire() as conn:
            return await recompute_member_centroid(conn, person_id)

    async def list_members(self) -> list[MemberInfo]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_MEMBERS)
        return [_member_info(r) for r in rows]

    async def get_member(self, person_id: str) -> MemberInfo | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_MEMBER, person_id)
        return _member_info(row) if row else None

    async def remove_member(self, person_id: str) -> bool:
        async with self._pool.acquire() as conn:
            # ON DELETE CASCADE removes the member's embeddings and centroid.
            result = await conn.execute("DELETE FROM members WHERE person_id = $1", person_id)
        return bool(result != "DELETE 0")

    async def member_count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) AS cnt FROM members")
        return int(row["cnt"])

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT person_id, embedding FROM embeddings ORDER BY person_id, created_at, id"
            )
        gallery: dict[str, list[np.ndarray]] = {}
        for r in rows:
            gallery.setdefault(r["person_id"], []).append(
                np.asarray(r["embedding"], dtype=np.float32)
            )
        return gallery


@dataclass
class _Member:
    name: str
    created_at: datetime


class InMemoryEnrollmentRepository:
    """Dict-backed gallery for unit tests: brute-force cosine, same centroid rule as Postgres.

    ``list_members`` orders names with Python string ordering, which matches Postgres collation
    for plain ASCII names (what the parity test uses) but not necessarily for accented or mixed
    case names.
    """

    def __init__(self) -> None:
        self._members: dict[str, _Member] = {}
        self._embeddings: dict[str, list[np.ndarray]] = {}
        self._centroids: dict[str, np.ndarray] = {}

    async def nearest_centroids(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        query = np.asarray(embedding, dtype=np.float64)
        hits = [
            CentroidHit(
                person_id=person_id,
                name=self._members[person_id].name,
                similarity=float(
                    np.dot(centroid, query) / (np.linalg.norm(centroid) * np.linalg.norm(query))
                ),
            )
            for person_id, centroid in self._centroids.items()
            if person_id in self._members
        ]
        hits.sort(key=lambda hit: (-hit.similarity, hit.person_id))
        return hits[:k]

    async def member_exists(self, person_id: str) -> bool:
        return person_id in self._members

    async def embedding_count(self, person_id: str) -> int:
        return len(self._embeddings.get(person_id, []))

    async def upsert_member(self, person_id: str, name: str) -> UpsertStatus:
        existing = self._members.get(person_id)
        if existing is not None:
            existing.name = name
            return "updated"
        self._members[person_id] = _Member(name=name, created_at=datetime.now(UTC))
        return "enrolled"

    async def insert_embeddings(self, person_id: str, embeddings: Sequence[np.ndarray]) -> None:
        self._embeddings.setdefault(person_id, []).extend(
            np.asarray(e, dtype=np.float32) for e in embeddings
        )

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        centroid = np.mean(self._embeddings[person_id], axis=0)
        centroid = (centroid / np.linalg.norm(centroid)).astype(np.float32)
        self._centroids[person_id] = centroid
        return centroid

    async def list_members(self) -> list[MemberInfo]:
        return [
            self._info(person_id)
            for person_id in sorted(self._members, key=lambda pid: self._members[pid].name)
        ]

    async def get_member(self, person_id: str) -> MemberInfo | None:
        return self._info(person_id) if person_id in self._members else None

    async def remove_member(self, person_id: str) -> bool:
        if person_id not in self._members:
            return False
        del self._members[person_id]
        self._embeddings.pop(person_id, None)
        self._centroids.pop(person_id, None)
        return True

    async def member_count(self) -> int:
        return len(self._members)

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        return {
            person_id: list(embeddings)
            for person_id, embeddings in sorted(self._embeddings.items())
            if embeddings
        }

    def _info(self, person_id: str) -> MemberInfo:
        member = self._members[person_id]
        return MemberInfo(
            person_id=person_id,
            name=member.name,
            embedding_count=len(self._embeddings.get(person_id, [])),
            created_at=member.created_at,
        )
