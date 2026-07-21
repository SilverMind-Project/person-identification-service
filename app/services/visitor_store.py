"""Visitor cluster store: persists unmatched face embeddings, clusters them
across visits, and exposes review/naming operations (identity-continuity M06).

Only NAMED clusters ever produce a household member; unnamed clusters exist
solely for caregiver review and never influence identification or tracking.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import asyncpg
import cv2
import numpy as np

from app import config
from app.models.visitor import (
    NameClusterResponse,
    VisitorClusterDetail,
    VisitorClusterSummary,
    VisitorSighting,
)
from app.observability import metrics
from app.services.centroid import insert_embeddings, recompute_member_centroid
from app.services.face_engine import FaceEngine
from app.services.face_models import DetectedFace, IdentifyResult
from app.services.minio_client import MinioClient

logger = logging.getLogger(__name__)

_UNNAMED_STATUSES = ("candidate", "surfaced", "dismissed")
# Clusters eligible to absorb a new sighting: everything except 'named' (a
# named cluster's biometric identity now belongs to a real member and must
# not keep absorbing anonymous sightings). Dismissed clusters are included
# deliberately: they keep absorbing matching sightings (so the same courier
# does not resurface weekly as a new cluster) but the promotion check below
# never flips a dismissed cluster back to 'surfaced'.
_JOINABLE_STATUSES = _UNNAMED_STATUSES
_RECENT_CROPS_LIMIT = 5
_RECENT_SIGHTINGS_LIMIT = 20


@dataclass
class _JoinableCluster:
    cluster_id: str
    centroid: np.ndarray
    sighting_count: int
    status: str
    first_seen_at: datetime
    last_seen_at: datetime


class VisitorStore:
    """Manages the visitor-cluster dataset: ingest, clustering, review, retention."""

    def __init__(self, pool: asyncpg.Pool, minio_client: MinioClient) -> None:
        self._pool = pool
        self._minio = minio_client

        self._clustering_enabled = bool(config.get("visitors.clustering_enabled", True))
        self._min_detection_score = float(config.get("visitors.min_detection_score", 0.6))
        self._min_face_px = float(config.get("visitors.min_face_px", 64))
        self._member_margin = float(config.get("visitors.member_margin", 0.05))
        self._cluster_join_threshold = float(config.get("visitors.cluster_join_threshold", 0.55))
        self._surface_min_days = int(config.get("visitors.surface_min_days", 3))
        self._surface_window_days = int(config.get("visitors.surface_window_days", 30))
        self._unnamed_retention_days = int(config.get("visitors.unnamed_retention_days", 60))
        self._enroll_top_k = int(config.get("visitors.enroll_top_k", 5))
        self._recognition_threshold = float(config.get("recognition.threshold", 0.4))

    @property
    def clustering_enabled(self) -> bool:
        return self._clustering_enabled

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def record_sighting(
        self,
        image: np.ndarray,
        face: DetectedFace,
        identity: IdentifyResult,
        seen_at: datetime | None = None,
        source: str = "identify",
    ) -> str | None:
        """Persist an unmatched face as a visitor sighting, if all gates pass.

        Returns the cluster_id the sighting joined or created, or None if the
        sighting was rejected (or clustering is disabled).
        """
        if not self._clustering_enabled:
            return None

        if face.det_score < self._min_detection_score:
            metrics.visitor_sightings_rejected_total.labels(reason="low_detection_score").inc()
            return None

        x1, y1, x2, y2 = face.bbox
        if min(x2 - x1, y2 - y1) < self._min_face_px:
            metrics.visitor_sightings_rejected_total.labels(reason="face_too_small").inc()
            return None

        # Fail closed: a borderline household match must never seed a visitor
        # cluster (privacy/correctness gate, D7).
        if identity.similarity >= (self._recognition_threshold - self._member_margin):
            metrics.visitor_sightings_rejected_total.labels(reason="near_member").inc()
            return None

        seen_at = seen_at or datetime.now(UTC)
        embedding = np.asarray(face.embedding, dtype=np.float32)

        async with self._pool.acquire() as conn, conn.transaction():
            cluster_id, sighting_id = await self._assign_and_insert(
                conn, embedding, float(face.det_score), seen_at, source
            )

        metrics.visitor_sightings_recorded_total.inc()
        await self._refresh_cluster_gauge()

        crop_object = self._crop_and_upload(image, face.bbox, cluster_id, sighting_id)
        if crop_object is not None:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "UPDATE visitor_sightings SET crop_object = $1 WHERE id = $2",
                    crop_object,
                    sighting_id,
                )

        return cluster_id

    async def _assign_and_insert(
        self,
        conn: asyncpg.Connection,
        embedding: np.ndarray,
        quality: float,
        seen_at: datetime,
        source: str,
    ) -> tuple[str, int]:
        """Join an existing open cluster or create a new one; insert the sighting.

        Returns (cluster_id, sighting_id).
        """
        open_clusters = await self._load_joinable_clusters(conn)

        best: _JoinableCluster | None = None
        best_sim = -1.0
        for cluster in open_clusters:
            sim = FaceEngine.compute_similarity(embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best = cluster

        if best is not None and best_sim >= self._cluster_join_threshold:
            cluster_id = best.cluster_id
            prev_count = best.sighting_count
            prev_centroid = best.centroid
        else:
            cluster_id = str(uuid.uuid4())
            prev_count = 0
            prev_centroid = embedding
            await conn.execute(
                """INSERT INTO visitor_clusters
                       (cluster_id, status, centroid, sighting_count, distinct_days,
                        first_seen_at, last_seen_at)
                   VALUES ($1, 'candidate', $2, 0, 0, $3, $3)""",
                cluster_id,
                embedding,
                seen_at,
            )

        sighting_row = await conn.fetchrow(
            """INSERT INTO visitor_sightings (cluster_id, embedding, quality, seen_at, source)
               VALUES ($1, $2, $3, $4, $5)
               RETURNING id""",
            cluster_id,
            embedding,
            quality,
            seen_at,
            source,
        )
        sighting_id = int(sighting_row["id"])

        new_count = prev_count + 1
        new_centroid = (prev_centroid * prev_count + embedding) / new_count
        new_centroid = new_centroid / np.linalg.norm(new_centroid)

        window_start = datetime.now(UTC) - timedelta(days=self._surface_window_days)
        distinct_days_row = await conn.fetchrow(
            """SELECT count(DISTINCT date_trunc('day', seen_at)) AS distinct_days
               FROM visitor_sightings
               WHERE cluster_id = $1 AND seen_at >= $2""",
            cluster_id,
            window_start,
        )
        distinct_days = int(distinct_days_row["distinct_days"])

        await conn.execute(
            """UPDATE visitor_clusters
               SET centroid = $2,
                   sighting_count = $3,
                   distinct_days = $4,
                   first_seen_at = LEAST(first_seen_at, $5),
                   last_seen_at = GREATEST(last_seen_at, $5),
                   updated_at = now(),
                   status = CASE
                       WHEN status = 'candidate' AND $4::int >= $6::int THEN 'surfaced'
                       ELSE status
                   END
               WHERE cluster_id = $1""",
            cluster_id,
            new_centroid,
            new_count,
            distinct_days,
            seen_at,
            self._surface_min_days,
        )

        return cluster_id, sighting_id

    async def _load_joinable_clusters(self, conn: asyncpg.Connection) -> list[_JoinableCluster]:
        rows = await conn.fetch(
            """SELECT cluster_id, centroid, sighting_count, status, first_seen_at, last_seen_at
               FROM visitor_clusters
               WHERE status = ANY($1::text[])""",
            list(_JOINABLE_STATUSES),
        )
        return [
            _JoinableCluster(
                cluster_id=str(r["cluster_id"]),
                centroid=np.asarray(r["centroid"], dtype=np.float32),
                sighting_count=r["sighting_count"],
                status=r["status"],
                first_seen_at=r["first_seen_at"],
                last_seen_at=r["last_seen_at"],
            )
            for r in rows
        ]

    def _crop_and_upload(
        self, image: np.ndarray, bbox: list[float], cluster_id: str, sighting_id: int
    ) -> str | None:
        """Crop the aligned face region and upload it to MinIO. Best-effort."""
        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            xi1, yi1 = max(0, int(x1)), max(0, int(y1))
            xi2, yi2 = min(w, int(x2)), min(h, int(y2))
            if xi2 <= xi1 or yi2 <= yi1:
                return None
            crop = image[yi1:yi2, xi1:xi2]

            ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return None

            object_name = f"visitor-crops/{cluster_id}/{sighting_id}.jpg"
            self._minio.upload_bytes(buf.tobytes(), object_name, content_type="image/jpeg")
            return object_name
        except Exception:
            logger.exception(
                "Failed to save visitor crop cluster_id=%s sighting_id=%s", cluster_id, sighting_id
            )
            return None

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    async def list_clusters(self, status: str | None = None) -> list[VisitorClusterSummary]:
        if not self._clustering_enabled:
            return []

        async with self._pool.acquire() as conn:
            if status is not None:
                rows = await conn.fetch(
                    """SELECT * FROM visitor_clusters WHERE status = $1
                       ORDER BY last_seen_at DESC""",
                    status,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM visitor_clusters ORDER BY last_seen_at DESC"
                )

            summaries = []
            for row in rows:
                crop_rows = await conn.fetch(
                    """SELECT crop_object FROM visitor_sightings
                       WHERE cluster_id = $1 AND crop_object IS NOT NULL
                       ORDER BY seen_at DESC LIMIT $2""",
                    row["cluster_id"],
                    _RECENT_CROPS_LIMIT,
                )
                summaries.append(self._to_summary(row, [r["crop_object"] for r in crop_rows]))
            return summaries

    async def get_cluster(self, cluster_id: str) -> VisitorClusterDetail | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM visitor_clusters WHERE cluster_id = $1", cluster_id
            )
            if row is None:
                return None

            sighting_rows = await conn.fetch(
                """SELECT seen_at, quality, crop_object FROM visitor_sightings
                   WHERE cluster_id = $1 ORDER BY seen_at DESC LIMIT $2""",
                cluster_id,
                _RECENT_SIGHTINGS_LIMIT,
            )
            crop_rows = [r["crop_object"] for r in sighting_rows if r["crop_object"] is not None][
                :_RECENT_CROPS_LIMIT
            ]

        summary = self._to_summary(row, crop_rows)
        return VisitorClusterDetail(
            **summary.model_dump(),
            recent_sightings=[
                VisitorSighting(seen_at=r["seen_at"], quality=r["quality"], crop_object=r["crop_object"])
                for r in sighting_rows
            ],
        )

    @staticmethod
    def _to_summary(row: asyncpg.Record, recent_crop_keys: list[str]) -> VisitorClusterSummary:
        return VisitorClusterSummary(
            cluster_id=str(row["cluster_id"]),
            status=row["status"],
            display_hint=row["display_hint"],
            named_person_id=row["named_person_id"],
            sighting_count=row["sighting_count"],
            distinct_days=row["distinct_days"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            recent_crop_keys=recent_crop_keys,
        )

    async def name_cluster(
        self, cluster_id: str, person_id: str, name: str
    ) -> NameClusterResponse:
        """Name a cluster into a household member. Raises ValueError on conflict,
        LookupError if the cluster does not exist."""
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(
                "SELECT * FROM visitor_clusters WHERE cluster_id = $1 FOR UPDATE", cluster_id
            )
            if row is None:
                raise LookupError(f"cluster '{cluster_id}' not found")
            if row["status"] == "named":
                if row["named_person_id"] != person_id:
                    raise ValueError(f"cluster '{cluster_id}' is already named")
                # Idempotent retry of a naming call that already committed: the
                # caller's BFF naming transaction must
                # be able to retry safely after a downstream failure without
                # this being treated as a conflict. Return the existing state
                # rather than re-inserting embeddings or touching the cluster row.
                member_row = await conn.fetchrow(
                    "SELECT name FROM members WHERE person_id = $1", person_id
                )
                embedding_count = await conn.fetchval(
                    "SELECT count(*) FROM embeddings WHERE person_id = $1", person_id
                )
                return NameClusterResponse(
                    cluster_id=cluster_id,
                    status="named",
                    named_person_id=person_id,
                    member_name=member_row["name"] if member_row else name,
                    embedding_count=embedding_count or 0,
                )

            existing = await conn.fetchrow(
                "SELECT 1 FROM members WHERE person_id = $1", person_id
            )
            if existing is not None:
                raise ValueError(f"person_id '{person_id}' already exists")

            embedding_rows = await conn.fetch(
                """SELECT embedding FROM visitor_sightings WHERE cluster_id = $1
                   ORDER BY quality DESC LIMIT $2""",
                cluster_id,
                self._enroll_top_k,
            )
            embeddings = [np.asarray(r["embedding"], dtype=np.float32) for r in embedding_rows]

            await conn.execute(
                "INSERT INTO members (person_id, name) VALUES ($1, $2)", person_id, name
            )
            await insert_embeddings(conn, person_id, embeddings)
            await recompute_member_centroid(conn, person_id)

            await conn.execute(
                """UPDATE visitor_clusters
                   SET status = 'named', named_person_id = $2, updated_at = now()
                   WHERE cluster_id = $1""",
                cluster_id,
                person_id,
            )

        metrics.visitor_named_total.inc()
        await self._refresh_cluster_gauge()

        return NameClusterResponse(
            cluster_id=cluster_id,
            status="named",
            named_person_id=person_id,
            member_name=name,
            embedding_count=len(embeddings),
        )

    async def dismiss_cluster(self, cluster_id: str) -> bool:
        """Mark a cluster dismissed. Returns False if not found.

        Raises ValueError if the cluster is already named (a real member's
        evidence cannot be dismissed)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT status FROM visitor_clusters WHERE cluster_id = $1", cluster_id
            )
            if row is None:
                return False
            if row["status"] == "named":
                raise ValueError(f"cluster '{cluster_id}' is already named")

            await conn.execute(
                "UPDATE visitor_clusters SET status = 'dismissed', updated_at = now() "
                "WHERE cluster_id = $1",
                cluster_id,
            )
        await self._refresh_cluster_gauge()
        return True

    async def merge_clusters(self, cluster_a: str, cluster_b: str) -> VisitorClusterSummary:
        """Merge two clusters that represent the same physical visitor.

        The older cluster (by created_at) is kept; centroid is recomputed
        over the union of sightings. Raises LookupError/ValueError on
        missing/named clusters."""
        if cluster_a == cluster_b:
            raise ValueError("cannot merge a cluster with itself")

        async with self._pool.acquire() as conn, conn.transaction():
            rows = await conn.fetch(
                """SELECT * FROM visitor_clusters WHERE cluster_id = ANY($1::uuid[])
                   FOR UPDATE""",
                [cluster_a, cluster_b],
            )
            by_id = {str(r["cluster_id"]): r for r in rows}
            if cluster_a not in by_id or cluster_b not in by_id:
                raise LookupError("both clusters must exist to merge")
            if by_id[cluster_a]["status"] == "named" or by_id[cluster_b]["status"] == "named":
                raise ValueError("cannot merge a named cluster")

            row_a, row_b = by_id[cluster_a], by_id[cluster_b]
            keep, other = (row_a, row_b) if row_a["created_at"] <= row_b["created_at"] else (
                row_b,
                row_a,
            )
            keep_id, other_id = str(keep["cluster_id"]), str(other["cluster_id"])

            await conn.execute(
                "UPDATE visitor_sightings SET cluster_id = $1 WHERE cluster_id = $2",
                keep_id,
                other_id,
            )

            embedding_rows = await conn.fetch(
                "SELECT embedding FROM visitor_sightings WHERE cluster_id = $1", keep_id
            )
            all_embeddings = [np.asarray(r["embedding"], dtype=np.float32) for r in embedding_rows]
            centroid = np.mean(all_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            window_start = datetime.now(UTC) - timedelta(days=self._surface_window_days)
            distinct_days_row = await conn.fetchrow(
                """SELECT count(DISTINCT date_trunc('day', seen_at)) AS distinct_days
                   FROM visitor_sightings WHERE cluster_id = $1 AND seen_at >= $2""",
                keep_id,
                window_start,
            )
            distinct_days = int(distinct_days_row["distinct_days"])

            new_first = min(keep["first_seen_at"], other["first_seen_at"])
            new_last = max(keep["last_seen_at"], other["last_seen_at"])
            new_count = keep["sighting_count"] + other["sighting_count"]
            new_status = keep["status"]
            if new_status == "candidate" and distinct_days >= self._surface_min_days:
                new_status = "surfaced"

            await conn.execute(
                """UPDATE visitor_clusters
                   SET centroid = $2, sighting_count = $3, distinct_days = $4,
                       first_seen_at = $5, last_seen_at = $6, status = $7, updated_at = now()
                   WHERE cluster_id = $1""",
                keep_id,
                centroid,
                new_count,
                distinct_days,
                new_first,
                new_last,
                new_status,
            )
            await conn.execute("DELETE FROM visitor_clusters WHERE cluster_id = $1", other_id)

            merged_row = await conn.fetchrow(
                "SELECT * FROM visitor_clusters WHERE cluster_id = $1", keep_id
            )

        await self._refresh_cluster_gauge()
        return self._to_summary(merged_row, [])

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    async def purge_expired(self) -> int:
        """Delete unnamed clusters (and their sightings/crops) past retention.

        Named clusters are exempt. Returns the number of clusters purged.
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._unnamed_retention_days)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT cluster_id FROM visitor_clusters
                   WHERE status = ANY($1::text[]) AND last_seen_at < $2""",
                list(_UNNAMED_STATUSES),
                cutoff,
            )
            cluster_ids = [str(r["cluster_id"]) for r in rows]

            for cluster_id in cluster_ids:
                crop_rows = await conn.fetch(
                    """SELECT crop_object FROM visitor_sightings
                       WHERE cluster_id = $1 AND crop_object IS NOT NULL""",
                    cluster_id,
                )
                for r in crop_rows:
                    try:
                        self._minio.delete_object(r["crop_object"])
                    except Exception:
                        logger.warning(
                            "Failed to delete visitor crop during purge: %s", r["crop_object"]
                        )

                await conn.execute("DELETE FROM visitor_clusters WHERE cluster_id = $1", cluster_id)
                metrics.visitor_clusters_purged_total.inc()

        if cluster_ids:
            logger.info("Purged %d expired unnamed visitor clusters", len(cluster_ids))
            await self._refresh_cluster_gauge()
        return len(cluster_ids)

    async def _refresh_cluster_gauge(self) -> None:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT status, count(*) AS cnt FROM visitor_clusters GROUP BY status"
            )
        counts = {r["status"]: r["cnt"] for r in rows}
        for status in ("candidate", "surfaced", "named", "dismissed"):
            metrics.visitor_clusters_total.labels(status=status).set(counts.get(status, 0))
