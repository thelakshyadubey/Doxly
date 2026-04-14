"""
app/api/routes/health.py
=========================
Liveness and readiness probe endpoints.

GET /health/live   → 200 always (process is alive)
GET /health/ready  → 200 if all dependencies healthy, 503 if any fail
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_neo4j, get_qdrant, get_redis
from app.models.api import LivenessResponse, ReadinessResponse
from app.stores.neo4j_store import Neo4jStore
from app.stores.qdrant_store import QdrantStore
from app.stores.redis_store import RedisStore
from app.utils.logger import get_logger

router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 immediately. Indicates the process is running.",
)
async def liveness() -> LivenessResponse:
    """Always returns ``{status: 'ok'}``."""
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Checks Qdrant, Neo4j, and Redis connectivity. Returns 503 if any dependency is unhealthy.",
)
async def readiness(
    response: Response,
    qdrant: QdrantStore = Depends(get_qdrant),
    neo4j: Neo4jStore = Depends(get_neo4j),
    redis: RedisStore = Depends(get_redis),
) -> ReadinessResponse:
    """
    Probe all three backing stores in parallel.

    Returns HTTP 503 if any dependency fails its ping.
    """
    qdrant_ok_task = asyncio.to_thread(qdrant.ping)
    neo4j_ok_task = neo4j.ping()
    redis_ok_task = redis.ping()

    qdrant_ok, neo4j_ok, redis_ok = await asyncio.gather(
        qdrant_ok_task, neo4j_ok_task, redis_ok_task
    )

    details: dict = {}
    if not qdrant_ok:
        details["qdrant"] = "ping failed"
    if not neo4j_ok:
        details["neo4j"] = "ping failed"
    if not redis_ok:
        details["redis"] = "ping failed"

    all_healthy = qdrant_ok and neo4j_ok and redis_ok
    http_status = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    response.status_code = http_status

    result = ReadinessResponse(
        status="ok" if all_healthy else "degraded",
        qdrant=qdrant_ok,
        neo4j=neo4j_ok,
        redis=redis_ok,
        details=details,
    )

    if not all_healthy:
        await logger.awarning("health.readiness_degraded", details=details)

    return result
