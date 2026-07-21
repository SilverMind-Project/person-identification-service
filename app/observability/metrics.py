"""Prometheus metrics for visitor clustering (identity-continuity M06).

Registered against the default ``prometheus_client`` registry so the existing
``GET /metrics`` route in ``app.routers.health`` exposes them without extra
plumbing.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge

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
