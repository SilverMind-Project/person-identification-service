"""Enrollment store: manages face embedding gallery with SQLite metadata and .npy files."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app import config
from app.models.enrollment import EnrollResult, MemberInfo
from app.services.face_engine import DetectedFace, FaceEngine, IdentifyResult

logger = logging.getLogger(__name__)


class EnrollmentStore:
    """Manages a gallery of enrolled household members with face embeddings."""

    def __init__(self, face_engine: FaceEngine) -> None:
        self._engine = face_engine
        self._db_path = config.get("storage.db_path", "data/face_db.sqlite")
        self._emb_dir = Path(config.get("storage.embeddings_dir", "data/embeddings"))
        self._threshold = float(config.get("recognition.threshold", 0.4))
        self._unknown_threshold = float(config.get("recognition.unknown_threshold", 0.25))
        self._emb_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # Cache centroids in memory for fast identification
        self._centroids: dict[str, tuple[str, np.ndarray]] = {}  # person_id -> (name, centroid)
        self._load_centroids()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS members (
                person_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding_count INTEGER NOT NULL DEFAULT 0,
                centroid_path TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_centroids(self) -> None:
        """Load all centroids into memory."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT person_id, name, centroid_path FROM members").fetchall()
        conn.close()

        self._centroids.clear()
        for person_id, name, centroid_path in rows:
            if centroid_path and Path(centroid_path).exists():
                centroid = np.load(centroid_path)
                self._centroids[person_id] = (name, centroid)
        logger.info("Loaded %d centroids into memory", len(self._centroids))

    def enroll(
        self, person_id: str, name: str, images: list[np.ndarray]
    ) -> EnrollResult:
        """Enroll a new person or add images to an existing member.

        Args:
            person_id: Unique identifier.
            name: Display name.
            images: List of BGR numpy arrays, each containing a face.

        Returns:
            EnrollResult with status and count of successful embeddings.
        """
        person_dir = self._emb_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        # Check if already enrolled
        conn = sqlite3.connect(self._db_path)
        try:
            existing = conn.execute(
                "SELECT embedding_count FROM members WHERE person_id = ?", (person_id,)
            ).fetchone()
            existing_count = existing[0] if existing else 0

            embeddings: list[np.ndarray] = []
            failed_indices: list[int] = []

            for idx, img in enumerate(images):
                faces = self._engine.detect_faces(img)
                if not faces:
                    failed_indices.append(idx)
                    continue
                # Use the largest face (by bbox area) if multiple detected
                best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                emb_path = person_dir / f"embedding_{existing_count + len(embeddings)}.npy"
                np.save(emb_path, best_face.embedding)
                embeddings.append(best_face.embedding)

            if not embeddings:
                return EnrollResult(
                    person_id=person_id,
                    name=name,
                    embedding_count=existing_count,
                    status="failed",
                    failed_images=failed_indices,
                )

            # Load existing embeddings to compute new centroid
            all_embeddings = list(embeddings)
            for i in range(existing_count):
                path = person_dir / f"embedding_{i}.npy"
                if path.exists():
                    all_embeddings.append(np.load(path))

            # Compute and save centroid
            centroid = np.mean(all_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # re-normalize
            centroid_path = person_dir / "centroid.npy"
            np.save(centroid_path, centroid)

            total_count = existing_count + len(embeddings)
            now = datetime.now(timezone.utc).isoformat()

            if existing:
                conn.execute(
                    "UPDATE members SET name = ?, embedding_count = ?, centroid_path = ? WHERE person_id = ?",
                    (name, total_count, str(centroid_path), person_id),
                )
                status = "updated"
            else:
                conn.execute(
                    "INSERT INTO members (person_id, name, created_at, embedding_count, centroid_path) VALUES (?, ?, ?, ?, ?)",
                    (person_id, name, now, total_count, str(centroid_path)),
                )
                status = "enrolled"

            conn.commit()
        finally:
            conn.close()

        # Update in-memory cache
        self._centroids[person_id] = (name, centroid)

        logger.info(
            "Enrolled person_id=%s name=%s embeddings=%d status=%s",
            person_id, name, total_count, status,
        )
        return EnrollResult(
            person_id=person_id,
            name=name,
            embedding_count=total_count,
            status=status,
            failed_images=failed_indices,
        )

    def identify(self, face: DetectedFace) -> IdentifyResult:
        """Identify a detected face against the enrolled gallery.

        Returns the best match or "unknown" if below threshold.
        """
        if not self._centroids:
            return IdentifyResult(
                person_id="unknown", name="Guest", confidence=0.0, bbox=face.bbox
            )

        best_id = "unknown"
        best_name = "Guest"
        best_score = -1.0

        for person_id, (name, centroid) in self._centroids.items():
            score = FaceEngine.compute_similarity(face.embedding, centroid)
            if score > best_score:
                best_score = score
                best_id = person_id
                best_name = name

        if best_score < self._threshold:
            return IdentifyResult(
                person_id="unknown", name="Guest", confidence=max(0.0, best_score), bbox=face.bbox
            )

        return IdentifyResult(
            person_id=best_id, name=best_name, confidence=best_score, bbox=face.bbox
        )

    def identify_all(self, faces: list[DetectedFace]) -> list[IdentifyResult]:
        """Identify all faces in a single frame."""
        return [self.identify(face) for face in faces]

    def remove_member(self, person_id: str) -> bool:
        """Remove a member and all their embeddings."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("DELETE FROM members WHERE person_id = ?", (person_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
        finally:
            conn.close()

        if not deleted:
            return False

        person_dir = self._emb_dir / person_id
        if person_dir.exists():
            shutil.rmtree(person_dir)

        self._centroids.pop(person_id, None)
        logger.info("Removed member person_id=%s", person_id)
        return True

    def list_members(self) -> list[MemberInfo]:
        """Return all enrolled members."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT person_id, name, embedding_count, created_at FROM members ORDER BY name"
        ).fetchall()
        conn.close()

        return [
            MemberInfo(
                person_id=r[0],
                name=r[1],
                embedding_count=r[2],
                created_at=datetime.fromisoformat(r[3]),
            )
            for r in rows
        ]

    def get_member(self, person_id: str) -> MemberInfo | None:
        conn = sqlite3.connect(self._db_path)
        row = conn.execute(
            "SELECT person_id, name, embedding_count, created_at FROM members WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        return MemberInfo(
            person_id=row[0],
            name=row[1],
            embedding_count=row[2],
            created_at=datetime.fromisoformat(row[3]),
        )

    @property
    def member_count(self) -> int:
        return len(self._centroids)
