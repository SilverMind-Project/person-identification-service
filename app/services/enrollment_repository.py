"""Persistence for the enrollment gallery: members, embeddings, centroids, and prototypes.

``EnrollmentRepository`` is the contract the enrollment and template services depend on. The
Postgres implementation holds every gallery query except the shared embedding/centroid/prototype
SQL in ``app.services.centroid`` (also used inside ``VisitorStore.name_cluster``'s transaction).
The in-memory peer is kept honest by a parity test against Postgres.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, Protocol

import asyncpg
import numpy as np

from app.models.enrollment import MemberInfo
from app.services.centroid import (
    insert_embeddings,
    rebuild_member_prototypes,
    recompute_member_centroid,
    replace_member_prototypes,
)
from app.services.prototype_builder import (
    EmbeddingSource,
    MemberEmbedding,
    PoseBin,
    Prototype,
    PrototypePolicy,
    build_prototypes,
)

UpsertStatus = Literal["enrolled", "updated"]


@dataclass(frozen=True)
class CentroidHit:
    """One enrolled member and its cosine similarity to a query embedding."""

    person_id: str
    name: str
    # Cosine similarity, 1 - pgvector cosine distance, in [-1, 1].
    similarity: float
    # The member's live gallery revision at query time (used for fail-closed calibration coverage).
    gallery_revision: int = 1


class EnrollmentRepository(Protocol):
    async def nearest_centroids(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        """The ``k`` member centroids closest to ``embedding``, most similar first."""
        ...

    async def nearest_identities(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        """The ``k`` identities with highest max-over-prototype similarity to ``embedding``."""
        ...

    async def member_exists(self, person_id: str) -> bool: ...

    async def embedding_count(self, person_id: str) -> int: ...

    async def upsert_member(self, person_id: str, name: str) -> UpsertStatus:
        """Create the member, or rename an existing one; report which happened."""
        ...

    async def insert_embeddings(
        self, person_id: str, embeddings: Sequence[np.ndarray | MemberEmbedding]
    ) -> None: ...

    async def add_member_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        pose_bin: PoseBin,
        source: EmbeddingSource,
        quality_score: float | None = None,
        quality_version: str | None = None,
        source_candidate_id: str | None = None,
        approved_by: str | None = None,
        approved_at: datetime | None = None,
        crop_object_key: str | None = None,
    ) -> int:
        """Append one provenance-tagged embedding and increment the member's gallery revision."""
        ...

    async def bump_gallery_revision(self, person_id: str) -> int:
        """Increment and return ``person_id``'s gallery revision."""
        ...

    async def get_gallery_revisions(self) -> dict[str, int]:
        """Map every enrolled ``person_id`` to its current gallery revision."""
        ...

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        """Set the member's centroid to the L2-normalized mean of all its embeddings."""
        ...

    async def replace_prototypes(self, person_id: str, prototypes: Sequence[Prototype]) -> None: ...

    async def list_prototypes(self, person_id: str) -> list[Prototype]: ...

    async def rebuild_prototypes(
        self, person_id: str, policy: PrototypePolicy = PrototypePolicy()
    ) -> list[Prototype]: ...

    async def delete_prototype(
        self,
        person_id: str,
        prototype_id: str,
        policy: PrototypePolicy = PrototypePolicy(),
    ) -> bool:
        """Delete one prototype's underlying embedding(s), rebuild centroid+prototypes, bump revision.

        Raises ``ValueError`` if deleting the prototype would leave the member with zero embeddings.
        """
        ...

    async def clear_crop_reference(self, crop_object_key: str) -> None:
        """Clear a deleted crop key from embeddings and prototypes while preserving the vectors."""
        ...

    async def promoted_candidate_ids(self) -> set[str]:
        """Return candidate ids already promoted into ``embeddings``."""
        ...

    async def backfill_missing_pose_bins(self, default_bin: PoseBin = "frontal") -> int:
        """Assign ``default_bin`` to embeddings whose ``pose_bin`` is NULL; return updated count."""
        ...

    async def list_members(self) -> list[MemberInfo]:
        """Every member with its embedding count, revision, and prototype counts, ordered by name."""
        ...

    async def get_member(self, person_id: str) -> MemberInfo | None: ...

    async def remove_member(self, person_id: str) -> bool:
        """Delete the member with its embeddings, prototypes, and centroid; False if missing."""
        ...

    async def member_count(self) -> int: ...

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        """Every member's enrollment embeddings, oldest first (for offline calibration)."""
        ...

    async def member_embedding_records(self) -> dict[str, list[MemberEmbedding]]:
        """Every member's embeddings with pose/quality/source metadata, oldest first."""
        ...


_SQL_NEAREST = """SELECT m.person_id, m.name, m.gallery_revision,
       1.0 - (c.centroid <=> $1::vector) AS similarity
FROM centroids c
JOIN members m ON m.person_id = c.person_id
ORDER BY c.centroid <=> $1::vector, m.person_id ASC
LIMIT $2"""

_SQL_NEAREST_IDENTITIES = """SELECT m.person_id, m.name, m.gallery_revision,
       MAX(1.0 - (p.embedding <=> $1::vector)) AS similarity
FROM face_prototypes p
JOIN members m ON m.person_id = p.person_id
GROUP BY m.person_id, m.name, m.gallery_revision
ORDER BY similarity DESC, m.person_id ASC
LIMIT $2"""

_SQL_MEMBERS = """SELECT m.person_id, m.name, m.created_at, m.gallery_revision,
       count(e.id)::int AS embedding_count
FROM members m
LEFT JOIN embeddings e ON e.person_id = m.person_id
GROUP BY m.person_id, m.name, m.created_at, m.gallery_revision
ORDER BY m.name"""

_SQL_MEMBER = """SELECT m.person_id, m.name, m.created_at, m.gallery_revision,
       count(e.id)::int AS embedding_count
FROM members m
LEFT JOIN embeddings e ON e.person_id = m.person_id
WHERE m.person_id = $1
GROUP BY m.person_id, m.name, m.created_at, m.gallery_revision"""

_SQL_PROTOTYPE_COUNTS = """SELECT person_id, pose_bin, count(*)::int AS cnt
FROM face_prototypes
GROUP BY person_id, pose_bin"""

_SQL_MEMBER_PROTOTYPE_COUNTS = """SELECT pose_bin, count(*)::int AS cnt
FROM face_prototypes
WHERE person_id = $1
GROUP BY pose_bin"""


def _member_info(row: asyncpg.Record, proto_counts: dict[str, int]) -> MemberInfo:
    return MemberInfo(
        person_id=row["person_id"],
        name=row["name"],
        embedding_count=row["embedding_count"],
        created_at=row["created_at"],
        gallery_revision=int(row["gallery_revision"]),
        prototype_counts=dict(sorted(proto_counts.items())),
    )


def _prototype_from_row(row: asyncpg.Record) -> Prototype:
    return Prototype(
        prototype_id=str(row["prototype_id"]),
        person_id=str(row["person_id"]),
        pose_bin=row["pose_bin"],
        embedding=np.asarray(row["embedding"], dtype=np.float32),
        member_count=int(row["member_count"]),
        source=row["source"],
        quality_score=None if row["quality_score"] is None else float(row["quality_score"]),
        quality_version=row["quality_version"],
        crop_object_key=row["crop_object_key"],
        source_embedding_id=(
            None if row["source_embedding_id"] is None else int(row["source_embedding_id"])
        ),
        approved_by=row["approved_by"],
        approved_at=row["approved_at"],
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
                gallery_revision=int(r["gallery_revision"]),
            )
            for r in rows
        ]

    async def nearest_identities(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_NEAREST_IDENTITIES, embedding, k)
        return [
            CentroidHit(
                person_id=str(r["person_id"]),
                name=str(r["name"]),
                similarity=float(r["similarity"]),
                gallery_revision=int(r["gallery_revision"]),
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

    async def insert_embeddings(
        self, person_id: str, embeddings: Sequence[np.ndarray | MemberEmbedding]
    ) -> None:
        async with self._pool.acquire() as conn:
            await insert_embeddings(conn, person_id, embeddings)

    async def add_member_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        pose_bin: PoseBin,
        source: EmbeddingSource,
        quality_score: float | None = None,
        quality_version: str | None = None,
        source_candidate_id: str | None = None,
        approved_by: str | None = None,
        approved_at: datetime | None = None,
        crop_object_key: str | None = None,
    ) -> int:
        record = MemberEmbedding(
            embedding=np.asarray(embedding, dtype=np.float32),
            pose_bin=pose_bin,
            source=source,
            quality_score=quality_score,
            quality_version=quality_version,
            source_candidate_id=source_candidate_id,
            approved_by=approved_by,
            approved_at=approved_at,
            crop_object_key=crop_object_key,
        )
        async with self._pool.acquire() as conn, conn.transaction():
            ids = await insert_embeddings(conn, person_id, [record])
            await conn.execute(
                "UPDATE members SET gallery_revision = gallery_revision + 1 WHERE person_id = $1",
                person_id,
            )
        return ids[0]

    async def bump_gallery_revision(self, person_id: str) -> int:
        async with self._pool.acquire() as conn:
            rev = await conn.fetchval(
                """UPDATE members SET gallery_revision = gallery_revision + 1
                   WHERE person_id = $1
                   RETURNING gallery_revision""",
                person_id,
            )
        if rev is None:
            raise LookupError(f"member '{person_id}' not found")
        return int(rev)

    async def get_gallery_revisions(self) -> dict[str, int]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT person_id, gallery_revision FROM members")
        return {str(r["person_id"]): int(r["gallery_revision"]) for r in rows}

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        async with self._pool.acquire() as conn:
            return await recompute_member_centroid(conn, person_id)

    async def replace_prototypes(self, person_id: str, prototypes: Sequence[Prototype]) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await replace_member_prototypes(conn, person_id, prototypes)

    async def list_prototypes(self, person_id: str) -> list[Prototype]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT prototype_id, person_id, pose_bin, embedding, member_count, source,
                          quality_score, quality_version, crop_object_key, source_embedding_id,
                          approved_by, approved_at, created_at
                   FROM face_prototypes
                   WHERE person_id = $1
                   ORDER BY pose_bin, quality_score DESC NULLS LAST, prototype_id""",
                person_id,
            )
        return [_prototype_from_row(r) for r in rows]

    async def rebuild_prototypes(
        self, person_id: str, policy: PrototypePolicy = PrototypePolicy()
    ) -> list[Prototype]:
        async with self._pool.acquire() as conn, conn.transaction():
            return await rebuild_member_prototypes(conn, person_id, policy)

    async def delete_prototype(
        self,
        person_id: str,
        prototype_id: str,
        policy: PrototypePolicy = PrototypePolicy(),
    ) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            proto = await conn.fetchrow(
                """SELECT pose_bin, source_embedding_id
                   FROM face_prototypes
                   WHERE person_id = $1 AND prototype_id = $2::uuid""",
                person_id,
                prototype_id,
            )
            if proto is None:
                return False

            source_emb_id = proto["source_embedding_id"]
            if source_emb_id is not None:
                remaining = int(
                    await conn.fetchval(
                        "SELECT count(*) FROM embeddings WHERE person_id = $1 AND id <> $2",
                        person_id,
                        int(source_emb_id),
                    )
                )
            else:
                remaining = int(
                    await conn.fetchval(
                        """SELECT count(*) FROM embeddings
                           WHERE person_id = $1 AND COALESCE(pose_bin, 'frontal') <> $2""",
                        person_id,
                        proto["pose_bin"],
                    )
                )
            if remaining < 1:
                raise ValueError(
                    "cannot remove the last embedding of a member; delete enrollment instead"
                )

            if source_emb_id is not None:
                await conn.execute(
                    "DELETE FROM embeddings WHERE person_id = $1 AND id = $2",
                    person_id,
                    int(source_emb_id),
                )
            else:
                await conn.execute(
                    """DELETE FROM embeddings
                       WHERE person_id = $1 AND COALESCE(pose_bin, 'frontal') = $2""",
                    person_id,
                    proto["pose_bin"],
                )
            await recompute_member_centroid(conn, person_id)
            await rebuild_member_prototypes(conn, person_id, policy)
            await conn.execute(
                "UPDATE members SET gallery_revision = gallery_revision + 1 WHERE person_id = $1",
                person_id,
            )
        return True

    async def clear_crop_reference(self, crop_object_key: str) -> None:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE embeddings SET crop_object_key = NULL WHERE crop_object_key = $1",
                crop_object_key,
            )
            await conn.execute(
                "UPDATE face_prototypes SET crop_object_key = NULL WHERE crop_object_key = $1",
                crop_object_key,
            )

    async def promoted_candidate_ids(self) -> set[str]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT DISTINCT source_candidate_id
                   FROM embeddings
                   WHERE source_candidate_id IS NOT NULL"""
            )
        return {str(r["source_candidate_id"]) for r in rows}

    async def backfill_missing_pose_bins(self, default_bin: PoseBin = "frontal") -> int:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE embeddings SET pose_bin = $1 WHERE pose_bin IS NULL",
                default_bin,
            )
        # asyncpg returns "UPDATE N"
        return int(result.split()[-1])

    async def list_members(self) -> list[MemberInfo]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SQL_MEMBERS)
            proto_rows = await conn.fetch(_SQL_PROTOTYPE_COUNTS)
        counts_by_person: dict[str, dict[str, int]] = {}
        for r in proto_rows:
            counts_by_person.setdefault(str(r["person_id"]), {})[str(r["pose_bin"])] = int(r["cnt"])
        return [_member_info(r, counts_by_person.get(str(r["person_id"]), {})) for r in rows]

    async def get_member(self, person_id: str) -> MemberInfo | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(_SQL_MEMBER, person_id)
            if not row:
                return None
            proto_rows = await conn.fetch(_SQL_MEMBER_PROTOTYPE_COUNTS, person_id)
        counts = {str(r["pose_bin"]): int(r["cnt"]) for r in proto_rows}
        return _member_info(row, counts)

    async def remove_member(self, person_id: str) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            # Visitor clusters attributed to this member lose their label and become unreviewed
            # again: the person they were judged to be no longer exists. The foreign key has no
            # delete action, so the DELETE below fails if this step is ever skipped.
            await conn.execute(
                """UPDATE visitor_clusters
                   SET status = 'candidate', attributed_person_id = NULL, reviewed_by = NULL,
                       reviewed_at = NULL, updated_at = now()
                   WHERE attributed_person_id = $1""",
                person_id,
            )
            # ON DELETE CASCADE removes the member's embeddings, face_prototypes, and centroid.
            result = await conn.execute("DELETE FROM members WHERE person_id = $1", person_id)
        return bool(result != "DELETE 0")

    async def member_count(self) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) AS cnt FROM members")
        return int(row["cnt"])

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        records = await self.member_embedding_records()
        return {pid: [rec.embedding for rec in items] for pid, items in records.items()}

    async def member_embedding_records(self) -> dict[str, list[MemberEmbedding]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, person_id, embedding, pose_bin, source, quality_score,
                          quality_version, source_candidate_id, approved_by, approved_at,
                          crop_object_key
                   FROM embeddings
                   ORDER BY person_id, created_at, id"""
            )
        gallery: dict[str, list[MemberEmbedding]] = {}
        for r in rows:
            gallery.setdefault(str(r["person_id"]), []).append(
                MemberEmbedding(
                    embedding=np.asarray(r["embedding"], dtype=np.float32),
                    embedding_id=int(r["id"]),
                    pose_bin=r["pose_bin"],
                    source=r["source"],
                    quality_score=None if r["quality_score"] is None else float(r["quality_score"]),
                    quality_version=r["quality_version"],
                    source_candidate_id=(
                        None if r["source_candidate_id"] is None else str(r["source_candidate_id"])
                    ),
                    approved_by=r["approved_by"],
                    approved_at=r["approved_at"],
                    crop_object_key=r["crop_object_key"],
                )
            )
        return gallery


@dataclass
class _Member:
    name: str
    created_at: datetime
    gallery_revision: int = 1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return float(np.dot(a64, b64) / (np.linalg.norm(a64) * np.linalg.norm(b64)))


class InMemoryEnrollmentRepository:
    """Dict-backed gallery for unit tests: brute-force cosine, same rules as Postgres.

    ``list_members`` orders names with Python string ordering, which matches Postgres collation
    for plain ASCII names (what the parity test uses) but not necessarily for accented or mixed
    case names.
    """

    def __init__(self) -> None:
        self._members: dict[str, _Member] = {}
        self._embeddings: dict[str, list[MemberEmbedding]] = {}
        self._centroids: dict[str, np.ndarray] = {}
        self._prototypes: dict[str, list[Prototype]] = {}
        self._next_embedding_id = 1

    async def nearest_centroids(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        hits = [
            CentroidHit(
                person_id=person_id,
                name=self._members[person_id].name,
                similarity=_cosine(centroid, embedding),
                gallery_revision=self._members[person_id].gallery_revision,
            )
            for person_id, centroid in self._centroids.items()
            if person_id in self._members
        ]
        hits.sort(key=lambda hit: (-hit.similarity, hit.person_id))
        return hits[:k]

    async def nearest_identities(self, embedding: np.ndarray, k: int) -> list[CentroidHit]:
        hits: list[CentroidHit] = []
        for person_id, protos in self._prototypes.items():
            if person_id not in self._members or not protos:
                continue
            best_sim = max(_cosine(p.embedding, embedding) for p in protos)
            hits.append(
                CentroidHit(
                    person_id=person_id,
                    name=self._members[person_id].name,
                    similarity=best_sim,
                    gallery_revision=self._members[person_id].gallery_revision,
                )
            )
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

    async def insert_embeddings(
        self, person_id: str, embeddings: Sequence[np.ndarray | MemberEmbedding]
    ) -> None:
        bucket = self._embeddings.setdefault(person_id, [])
        for item in embeddings:
            emb_id = self._next_embedding_id
            self._next_embedding_id += 1
            if isinstance(item, MemberEmbedding):
                bucket.append(
                    replace(
                        item,
                        embedding=np.asarray(item.embedding, dtype=np.float32),
                        embedding_id=emb_id,
                    )
                )
            else:
                bucket.append(
                    MemberEmbedding(
                        embedding=np.asarray(item, dtype=np.float32),
                        embedding_id=emb_id,
                    )
                )

    async def add_member_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        pose_bin: PoseBin,
        source: EmbeddingSource,
        quality_score: float | None = None,
        quality_version: str | None = None,
        source_candidate_id: str | None = None,
        approved_by: str | None = None,
        approved_at: datetime | None = None,
        crop_object_key: str | None = None,
    ) -> int:
        emb_id = self._next_embedding_id
        self._next_embedding_id += 1
        self._embeddings.setdefault(person_id, []).append(
            MemberEmbedding(
                embedding=np.asarray(embedding, dtype=np.float32),
                embedding_id=emb_id,
                pose_bin=pose_bin,
                source=source,
                quality_score=quality_score,
                quality_version=quality_version,
                source_candidate_id=source_candidate_id,
                approved_by=approved_by,
                approved_at=approved_at,
                crop_object_key=crop_object_key,
            )
        )
        if person_id in self._members:
            self._members[person_id].gallery_revision += 1
        return emb_id

    async def bump_gallery_revision(self, person_id: str) -> int:
        member = self._members.get(person_id)
        if member is None:
            raise LookupError(f"member '{person_id}' not found")
        member.gallery_revision += 1
        return member.gallery_revision

    async def get_gallery_revisions(self) -> dict[str, int]:
        return {pid: m.gallery_revision for pid, m in self._members.items()}

    async def recompute_centroid(self, person_id: str) -> np.ndarray:
        centroid = np.mean([e.embedding for e in self._embeddings[person_id]], axis=0)
        centroid = (centroid / np.linalg.norm(centroid)).astype(np.float32)
        self._centroids[person_id] = centroid
        return centroid

    async def replace_prototypes(self, person_id: str, prototypes: Sequence[Prototype]) -> None:
        now = datetime.now(UTC)
        self._prototypes[person_id] = [
            replace(p, person_id=person_id, created_at=p.created_at or now) for p in prototypes
        ]

    async def list_prototypes(self, person_id: str) -> list[Prototype]:
        protos = list(self._prototypes.get(person_id, []))
        protos.sort(
            key=lambda p: (
                p.pose_bin,
                -(p.quality_score if p.quality_score is not None else -1.0),
                p.prototype_id,
            )
        )
        return protos

    async def rebuild_prototypes(
        self, person_id: str, policy: PrototypePolicy = PrototypePolicy()
    ) -> list[Prototype]:
        built = build_prototypes(person_id, self._embeddings.get(person_id, []), policy)
        await self.replace_prototypes(person_id, built)
        return built

    async def delete_prototype(
        self,
        person_id: str,
        prototype_id: str,
        policy: PrototypePolicy = PrototypePolicy(),
    ) -> bool:
        protos = self._prototypes.get(person_id, [])
        target = next((p for p in protos if p.prototype_id == prototype_id), None)
        if target is None:
            return False
        current = self._embeddings.get(person_id, [])
        if target.source_embedding_id is not None:
            remaining = [e for e in current if e.embedding_id != target.source_embedding_id]
        else:
            remaining = [e for e in current if (e.pose_bin or "frontal") != target.pose_bin]
        if not remaining:
            raise ValueError(
                "cannot remove the last embedding of a member; delete enrollment instead"
            )
        self._embeddings[person_id] = remaining
        await self.recompute_centroid(person_id)
        await self.rebuild_prototypes(person_id, policy)
        if person_id in self._members:
            self._members[person_id].gallery_revision += 1
        return True

    async def clear_crop_reference(self, crop_object_key: str) -> None:
        for pid, items in self._embeddings.items():
            self._embeddings[pid] = [
                replace(e, crop_object_key=None) if e.crop_object_key == crop_object_key else e
                for e in items
            ]
        for pid, protos in self._prototypes.items():
            self._prototypes[pid] = [
                replace(p, crop_object_key=None) if p.crop_object_key == crop_object_key else p
                for p in protos
            ]

    async def promoted_candidate_ids(self) -> set[str]:
        return {
            e.source_candidate_id
            for items in self._embeddings.values()
            for e in items
            if e.source_candidate_id is not None
        }

    async def backfill_missing_pose_bins(self, default_bin: PoseBin = "frontal") -> int:
        updated = 0
        for pid, items in self._embeddings.items():
            new_items: list[MemberEmbedding] = []
            for e in items:
                if e.pose_bin is None:
                    new_items.append(replace(e, pose_bin=default_bin))
                    updated += 1
                else:
                    new_items.append(e)
            self._embeddings[pid] = new_items
        return updated

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
        self._prototypes.pop(person_id, None)
        return True

    async def member_count(self) -> int:
        return len(self._members)

    async def member_embeddings(self) -> dict[str, list[np.ndarray]]:
        return {
            person_id: [e.embedding for e in items]
            for person_id, items in sorted(self._embeddings.items())
            if items
        }

    async def member_embedding_records(self) -> dict[str, list[MemberEmbedding]]:
        return {
            person_id: list(items) for person_id, items in sorted(self._embeddings.items()) if items
        }

    def _info(self, person_id: str) -> MemberInfo:
        member = self._members[person_id]
        counts: dict[str, int] = {}
        for proto in self._prototypes.get(person_id, []):
            counts[proto.pose_bin] = counts.get(proto.pose_bin, 0) + 1
        return MemberInfo(
            person_id=person_id,
            name=member.name,
            embedding_count=len(self._embeddings.get(person_id, [])),
            created_at=member.created_at,
            gallery_revision=member.gallery_revision,
            prototype_counts=dict(sorted(counts.items())),
        )
