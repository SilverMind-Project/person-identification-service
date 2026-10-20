"""Prometheus metrics for face measurement and visitor clustering.

Registered against the default ``prometheus_client`` registry so the existing
``GET /metrics`` route in ``app.routers.health`` exposes them without extra
plumbing.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

face_interocular_px = Histogram(
    "pid_face_interocular_px",
    "Distance between the two eye keypoints of each identified face, in pixels of the submitted "
    "image. Recognition degrades sharply below roughly 16 to 24 px.",
    ["recognition_state"],
    buckets=(8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128),
)

face_quality_score = Histogram(
    "pid_face_quality_score",
    "Face image quality score in [0, 1] of each identified face (heuristic scorer: resolution, "
    "sharpness, detection, pitch).",
    ["recognition_state"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

visitor_sightings_recorded_total = Counter(
    "pid_visitor_sightings_recorded_total",
    "Unmatched face sightings persisted into the visitor cluster store.",
)

visitor_sightings_rejected_total = Counter(
    "pid_visitor_sightings_rejected_total",
    "Unmatched face sightings rejected before persistence, by reason.",
    ["reason"],
)

visitor_clusters_total = Gauge(
    "pid_visitor_clusters_total",
    "Visitor clusters currently in each status.",
    ["status"],
)

visitor_clusters_purged_total = Counter(
    "pid_visitor_clusters_purged_total",
    "Unnamed visitor clusters deleted by the retention job.",
)

visitor_named_total = Counter(
    "pid_visitor_named_total",
    "Visitor clusters named into a household member.",
)
