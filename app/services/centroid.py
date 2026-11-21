"""Shared embedding, centroid, and prototype persistence for the enrollment gallery.

Used by direct enrollment (``EnrollmentStore``), promotion, re-embedding, and visitor-cluster
naming (``VisitorStore.name_cluster``): every write path shares the same SQL so centroids and
pose-binned prototypes are never a function of which code path added the embeddings. Every read
and write is scoped to one ``recognizer_version`` (defaulting to ``DEFAULT_RECOGNIZER_VERSION``).
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
from app.services.recognizer_profile import DEFAULT_RECOGNIZER_VERSION


async def insert_embeddings(
    conn: asyncpg.Connection,
    person_id: str,
    embeddings: Sequence[np.ndarray | MemberEmbedding],
    *,
    recognizer_version: str = DEFAULT_RECOGNIZER_VERSION,
) -> list[int]:
    """Insert new embedding rows for *(person_id, recognizer_version)* and return their ids."""
    ids: list[int] = []
    for item in embeddings:
        if isinstance(item, MemberEmbedding):
            row = await conn.fetchrow(
                """INSERT INTO embeddings (
                       person_id, embedding, pose_bin, source, quality_score, quality_version,
                       source_candidate_id, approved_by, approved_at, crop_object_key,
                       recognizer_version
                   )
                   VALUES ($1, $2, $3, $4, $5, $6, $7::uuid, $8, $9, $10, $11)
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
                recognizer_version,
            )
        else:
            row = await conn.fetchrow(
                """INSERT INTO embeddings (person_id, embedding, recognizer_version)
                   VALUES ($1, $2, $3)
                   RETURNING id""",
                person_id,
                np.asarray(item, dtype=np.float32),
                recognizer_version,
            )
        ids.append(int(row["id"]))
    return ids


async def recompute_member_centroid(
    conn: asyncpg.Connection,
    person_id: str,
    *,
    recognizer_version: str = DEFAULT_RECOGNIZER_VERSION,
) -> np.ndarray:
    """Recompute and upsert *(person_id, recognizer_version)*'s centroid from its embeddings.

    Returns the new centroid (mean of all embeddings for that recognizer version, L2-normalized).
    """
    rows = await conn.fetch(
        """SELECT embedding FROM embeddings
           WHERE person_id = $1 AND recognizer_version = $2""",
        person_id,
        recognizer_version,
    )
    all_embeddings = [r["embedding"] for r in rows]
    centroid = np.mean(all_embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    await conn.execute(
        """INSERT INTO centroids (person_id, recognizer_version, centroid, updated_at)
           VALUES ($1, $2, $3, now())
           ON CONFLICT (person_id, recognizer_version)
           DO UPDATE SET centroid = excluded.centroid, updated_at = now()""",
        person_id,
        recognizer_version,
        centroid,
    )
    return centroid


async def fetch_member_embeddings(
    conn: asyncpg.Connection,
    person_id: str,
    *,
    recognizer_version: str = DEFAULT_RECOGNIZER_VERSION,
) -> list[MemberEmbedding]:
    """Read all embedding rows with provenance metadata for *(person_id, recognizer_version)*."""
    rows = await conn.fetch(
        """SELECT id, embedding, pose_bin, source, quality_score, quality_version,
                  source_candidate_id, approved_by, approved_at, crop_object_key,
                  recognizer_version
           FROM embeddings
           WHERE person_id = $1 AND recognizer_version = $2
           ORDER BY created_at, id""",
        person_id,
        recognizer_version,
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
            recognizer_version=str(r["recognizer_version"]),
        )
        for r in rows
    ]


async def replace_member_prototypes(
    conn: asyncpg.Connection,
    person_id: str,
    prototypes: Sequence[Prototype],
    *,
    recognizer_version: str = DEFAULT_RECOGNIZER_VERSION,
) -> None:
    """Replace *(person_id, recognizer_version)*'s rows in ``face_prototypes``."""
    await conn.execute(
        "DELETE FROM face_prototypes WHERE person_id = $1 AND recognizer_version = $2",
        person_id,
        recognizer_version,
    )
    for proto in prototypes:
        await conn.execute(
            """INSERT INTO face_prototypes (
                   prototype_id, person_id, pose_bin, embedding, member_count, source,
                   quality_score, quality_version, crop_object_key, source_embedding_id,
                   approved_by, approved_at, recognizer_version
               )
               VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)""",
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
            recognizer_version,
        )


async def rebuild_member_prototypes(
    conn: asyncpg.Connection,
    person_id: str,
    policy: PrototypePolicy = PrototypePolicy(),
    *,
    recognizer_version: str = DEFAULT_RECOGNIZER_VERSION,
) -> list[Prototype]:
    """Rebuild and persist pose-binned prototypes for *(person_id, recognizer_version)*."""
    items = await fetch_member_embeddings(conn, person_id, recognizer_version=recognizer_version)
    prototypes = build_prototypes(person_id, items, policy, recognizer_version=recognizer_version)
    await replace_member_prototypes(
        conn, person_id, prototypes, recognizer_version=recognizer_version
    )
    return prototypes
