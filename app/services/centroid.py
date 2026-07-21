"""Shared centroid computation for the enrollment gallery.

Used by both direct enrollment (``EnrollmentStore.enroll``) and visitor-cluster
naming (``VisitorStore.name_cluster``): both insert new embeddings for a
``person_id`` and must recompute the centroid identically (mean of all rows,
L2-normalized) so a member's centroid is never a function of which code path
added the images.
"""

from __future__ import annotations

import asyncpg
import numpy as np


async def insert_embeddings(
    conn: asyncpg.Connection, person_id: str, embeddings: list[np.ndarray]
) -> None:
    """Insert new embedding rows for *person_id*."""
    for emb in embeddings:
        await conn.execute(
            "INSERT INTO embeddings (person_id, embedding) VALUES ($1, $2)",
            person_id,
            emb,
        )


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
