"""Shared embedding, centroid, and prototype persistence for the enrollment gallery.

Used by direct enrollment (``EnrollmentStore``), promotion, and visitor-cluster naming
(``VisitorStore.name_cluster``): every write path shares the same SQL so centroids and
pose-binned prototypes are never a function of which code path added the embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence

import asyncpg
import numpy as np

from app.services.prototype_builder import (
    MemberEmbedding,
    Prototype,
    PrototypePolicy,
    build_prototypes,
)


async def insert_embeddings(
    conn: asyncpg.Connection,
    person_id: str,
    embeddings: Sequence[np.ndarray | MemberEmbedding],
) -> list[int]:
    """Insert new embedding rows for *person_id* and return their generated ids."""
    ids: list[int] = []
    for item in embeddings:
        if isinstance(item, MemberEmbedding):
            row = await conn.fetchrow(
                """INSERT INTO embeddings (
                       person_id, embedding, pose_bin, source, quality_score, quality_version,
                       source_candidate_id, approved_by, approved_at, crop_object_key
                   )
                   VALUES ($1, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10)
                   RETURNING id""",
                person_id,
                np.asarray(item.embedding, dtype=np.float32),
                item.pose_bin,
                item.source,
                item.quality_score,
                item.quality_version,
                item.source_candidate_id,
                item.approved_by,
                item.approved_at,
                item.crop_object_key,
            )
        else:
            row = await conn.fetchrow(
                "INSERT INTO embeddings (person_id, embedding) VALUES ($1, $2) RETURNING id",
                person_id,
                np.asarray(item, dtype=np.float32),
            )
        ids.append(int(row["id"]))
    return ids


async def recompute_member_centroid(conn: asyncpg.Connection, person_id: str) -> np.ndarray:
    """Recompute and upsert *person_id*'s centroid from all its embedding rows.

    Returns the new centroid (mean of all embeddings, L2-normalized).
    """
    rows = await conn.fetch("SELECT embedding FROM embeddings WHERE person_id = $1", person_id)
    all_embeddings = [r["embedding"] for r in rows]
    centroid = np.mean(all_embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    await conn.execute(
        """INSERT INTO centroids (person_id, centroid, updated_at)
           VALUES ($1, $2, now())
           ON CONFLICT (person_id)
           DO UPDATE SET centroid = excluded.centroid, updated_at = now()""",
        person_id,
        centroid,
    )
    return centroid


async def fetch_member_embeddings(
    conn: asyncpg.Connection, person_id: str
) -> list[MemberEmbedding]:
    """Read all embedding rows with provenance metadata for *person_id*, oldest first."""
    rows = await conn.fetch(
        """SELECT id, embedding, pose_bin, source, quality_score, quality_version,
                  source_candidate_id, approved_by, approved_at, crop_object_key
           FROM embeddings
           WHERE person_id = $1
           ORDER BY created_at, id""",
        person_id,
    )
    return [
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
        for r in rows
    ]


async def replace_member_prototypes(
    conn: asyncpg.Connection,
    person_id: str,
    prototypes: Sequence[Prototype],
) -> None:
    """Replace *person_id*'s rows in ``face_prototypes`` with *prototypes*."""
    await conn.execute("DELETE FROM face_prototypes WHERE person_id = $1", person_id)
    for proto in prototypes:
        await conn.execute(
            """INSERT INTO face_prototypes (
                   prototype_id, person_id, pose_bin, embedding, member_count, source,
                   quality_score, quality_version, crop_object_key, source_embedding_id,
                   approved_by, approved_at
               )
               VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
            proto.prototype_id,
            person_id,
            proto.pose_bin,
            np.asarray(proto.embedding, dtype=np.float32),
            proto.member_count,
            proto.source,
            proto.quality_score,
            proto.quality_version,
            proto.crop_object_key,
            proto.source_embedding_id,
            proto.approved_by,
            proto.approved_at,
        )


async def rebuild_member_prototypes(
    conn: asyncpg.Connection,
    person_id: str,
    policy: PrototypePolicy = PrototypePolicy(),
) -> list[Prototype]:
    """Rebuild and persist pose-binned prototypes for *person_id* from all its embeddings."""
    items = await fetch_member_embeddings(conn, person_id)
    prototypes = build_prototypes(person_id, items, policy)
    await replace_member_prototypes(conn, person_id, prototypes)
    return prototypes
