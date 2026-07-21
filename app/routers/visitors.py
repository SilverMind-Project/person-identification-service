"""Visitor cluster review endpoints (identity-continuity M06).

Naming is the single privileged transition here: it moves biometric data from
the visitor dataset into the enrollment (member) dataset. This router never
creates a CC ``household_members`` row; that boundary belongs to M07's BFF
orchestration, which calls both this API and CC's own member endpoint in one
caregiver action.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Request

from app.models.visitor import (
    NameClusterRequest,
    NameClusterResponse,
    VisitorClusterDetail,
    VisitorClusterListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/visitors", tags=["visitors"])

_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _require_enabled(store) -> None:
    if not store.clustering_enabled:
        raise HTTPException(status_code=409, detail="Visitor clustering is disabled")


@router.get("/clusters", response_model=VisitorClusterListResponse)
async def list_clusters(request: Request, status: str | None = None):
    store = request.app.state.visitor_store
    clusters = await store.list_clusters(status=status)
    return VisitorClusterListResponse(clusters=clusters, total=len(clusters))


@router.get("/clusters/{cluster_id}", response_model=VisitorClusterDetail)
async def get_cluster(request: Request, cluster_id: str):
    store = request.app.state.visitor_store
    detail = await store.get_cluster(cluster_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return detail


@router.post("/clusters/{cluster_id}/name", response_model=NameClusterResponse)
async def name_cluster(request: Request, cluster_id: str, body: NameClusterRequest):
    store = request.app.state.visitor_store
    _require_enabled(store)

    if not _SLUG_RE.match(body.person_id):
        raise HTTPException(
            status_code=400,
            detail="person_id must be a lowercase slug (letters, digits, hyphens)",
        )

    try:
        result = await store.name_cluster(cluster_id, body.person_id, body.name)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    logger.info(
        "POST /visitors/clusters/%s/name -> person_id=%s embeddings=%d",
        cluster_id,
        body.person_id,
        result.embedding_count,
    )
    return result


@router.post("/clusters/{cluster_id}/dismiss")
async def dismiss_cluster(request: Request, cluster_id: str):
    store = request.app.state.visitor_store
    _require_enabled(store)

    try:
        found = await store.dismiss_cluster(cluster_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not found:
        raise HTTPException(status_code=404, detail=f"Cluster '{cluster_id}' not found")
    return {"cluster_id": cluster_id, "status": "dismissed"}


@router.post("/clusters/{cluster_a}/merge/{cluster_b}")
async def merge_clusters(request: Request, cluster_a: str, cluster_b: str):
    store = request.app.state.visitor_store
    _require_enabled(store)

    try:
        merged = await store.merge_clusters(cluster_a, cluster_b)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return merged
