"""
app/models/api.py
=================
FastAPI request/response schemas.

These are kept separate from domain.py so that the public API surface can
evolve independently of internal domain models.  All schemas use Pydantic v2.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from backend.app.models.domain import AnswerResult, Citation, DocType, SessionStatus


# ──────────────────────────────────────────────────────────────────────────────
# Ingest — POST /ingest/upload
# ──────────────────────────────────────────────────────────────────────────────


class UploadResponse(BaseModel):
    """
    Returned immediately after a page image is uploaded.

    The pipeline (classification → coref → chunking → embedding) is NOT
    triggered yet — it runs when the session window closes or /flush is called.
    """

    session_id: str
    page_count: int = Field(..., description="Total pages accumulated in this session so far")
    status: SessionStatus = SessionStatus.QUEUED


# ──────────────────────────────────────────────────────────────────────────────
# Ingest — POST /ingest/flush/{session_id}
# ──────────────────────────────────────────────────────────────────────────────


class FlushResponse(BaseModel):
    """Returned after the full ingestion pipeline completes for a session."""

    session_id: str
    chunk_count: int = Field(..., description="Number of child chunks written to Qdrant")
    entity_count: int = Field(..., description="Distinct entities written to Neo4j")
    doc_type: DocType
    status: SessionStatus = SessionStatus.INDEXED


# ──────────────────────────────────────────────────────────────────────────────
# Query — POST /query
# ──────────────────────────────────────────────────────────────────────────────


class QueryFilters(BaseModel):
    """Optional narrowing filters applied during retrieval."""

    doc_type: Optional[DocType] = Field(
        default=None, description="Restrict retrieval to a specific document type"
    )
    session_id: Optional[str] = Field(
        default=None, description="Restrict retrieval to a specific session"
    )


class QueryRequest(BaseModel):
    """Body for POST /query."""

    user_id: str = Field(..., description="Opaque user identifier for tenant isolation")
    query: str = Field(..., min_length=1, description="Natural-language question")
    filters: QueryFilters = Field(default_factory=QueryFilters)


class QueryResponse(BaseModel):
    """Non-streaming query response, mirrors AnswerResult with API envelope."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────


class LivenessResponse(BaseModel):
    """GET /health/live — always returns ok."""

    status: str = "ok"


class ReadinessResponse(BaseModel):
    """GET /health/ready — reports per-dependency connectivity."""

    status: str = Field(..., description="'ok' if all dependencies are healthy, else 'degraded'")
    qdrant: bool
    neo4j: bool
    redis: bool
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-dependency error messages when not healthy",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Generic error envelope
# ──────────────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error body returned on 4xx / 5xx responses."""

    error: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable explanation")
    detail: Optional[Any] = Field(default=None, description="Optional structured debug info")
