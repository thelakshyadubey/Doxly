"""
app/api/routes/health.py
=========================
Liveness and readiness probe endpoints.

GET /health/live   → 200 always (process is alive)
GET /health/ready  → 200 if all dependencies healthy, 503 if any fail
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, Response, status

from backend.app.models.api import LivenessResponse, ReadinessResponse
from backend.app.utils.logger import get_logger

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
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """
    Probe all three backing stores in parallel.

    Reads stores directly from ``app.state`` so that a None store (startup
    failure) is reported as unhealthy rather than raising an unhandled 503
    before the health response body can be built.

    Returns HTTP 503 if any dependency is absent or fails its ping.
    """
    qdrant = request.app.state.qdrant_store
    neo4j = request.app.state.neo4j_store
    redis = request.app.state.redis_store

    async def _ping_qdrant() -> bool:
        if qdrant is None:
            return False
        return await asyncio.to_thread(qdrant.ping)

    async def _ping_neo4j() -> bool:
        if neo4j is None:
            return False
        return await neo4j.ping()

    async def _ping_redis() -> bool:
        if redis is None:
            return False
        return await redis.ping()

    qdrant_ok, neo4j_ok, redis_ok = await asyncio.gather(
        _ping_qdrant(), _ping_neo4j(), _ping_redis()
    )

    details: dict = {}
    if not qdrant_ok:
        details["qdrant"] = "unavailable" if qdrant is None else "ping failed"
    if not neo4j_ok:
        details["neo4j"] = "unavailable" if neo4j is None else "ping failed"
    if not redis_ok:
        details["redis"] = "unavailable" if redis is None else "ping failed"

    all_healthy = qdrant_ok and neo4j_ok and redis_ok
    response.status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

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
