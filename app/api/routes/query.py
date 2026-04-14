"""
app/api/routes/query.py
========================
QnA endpoints — non-streaming and SSE streaming.

POST /query
    Body: {user_id, query, filters?}
    Returns: AnswerResult (full answer + citations + confidence)

GET /query/stream
    Query params: user_id, query, filters (JSON-encoded string)
    Returns: StreamingResponse (text/event-stream)
    Each event: ``data: {token}\\n\\n``
    Final:      ``data: [DONE]\\n\\n``
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_gemini, get_gemini_embed, get_neo4j, get_qdrant
from app.config.settings import Settings, get_settings
from app.models.api import QueryFilters, QueryRequest, QueryResponse
from app.services.answer_service import AnswerService
from app.services.retrieval_service import RetrievalService
from app.stores.neo4j_store import Neo4jStore
from app.stores.qdrant_store import QdrantStore
from app.utils.logger import get_logger

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


# ── Helper: build services from app.state ─────────────────────────────────────


def _build_services(
    request: Request, settings: Settings
) -> tuple[RetrievalService, AnswerService]:
    """
    Compose ``RetrievalService`` and ``AnswerService`` from app.state singletons.

    Args:
        request:  FastAPI request carrying app.state.
        settings: Application settings.

    Returns:
        Tuple of ``(RetrievalService, AnswerService)``.
    """
    qdrant: QdrantStore = request.app.state.qdrant_store
    neo4j: Neo4jStore = request.app.state.neo4j_store
    gemini_model = request.app.state.gemini_model
    embed_model: str = request.app.state.gemini_embed_model

    retrieval = RetrievalService(
        qdrant_store=qdrant,
        neo4j_store=neo4j,
        embedding_model=embed_model,
        reasoning_model=gemini_model,
        retrieval_top_k=settings.retrieval_top_k,
        final_top_k=settings.final_top_k,
    )
    answer = AnswerService(model=gemini_model)
    return retrieval, answer


# ── POST /query ───────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=QueryResponse,
    summary="Non-streaming document QnA",
    description="Retrieve relevant chunks via hybrid search and generate an answer using Gemini.",
)
async def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """
    Hybrid retrieval + Gemini answer generation (non-streaming).

    Args:
        body:     ``QueryRequest`` with user_id, query string, and optional filters.
        request:  FastAPI request (carries app.state).
        settings: Application settings (injected).

    Returns:
        ``QueryResponse`` with answer, citations, and confidence.
    """
    retrieval_svc, answer_svc = _build_services(request, settings)

    doc_type = body.filters.doc_type.value if body.filters.doc_type else None
    session_id = body.filters.session_id

    try:
        ranked_chunks = await retrieval_svc.retrieve(
            query=body.query,
            user_id=body.user_id,
            doc_type=doc_type,
            session_id=session_id,
        )
    except Exception as exc:
        await logger.aerror("query.retrieval_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )

    try:
        result = await answer_svc.answer(body.query, ranked_chunks)
    except Exception as exc:
        await logger.aerror("query.answer_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation failed: {exc}",
        )

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
    )


# ── GET /query/stream ─────────────────────────────────────────────────────────


@router.get(
    "/stream",
    summary="Streaming document QnA (SSE)",
    description=(
        "Retrieve relevant chunks and stream the Gemini answer as Server-Sent Events. "
        "Each event contains one or more tokens. Final event is ``data: [DONE]``."
    ),
)
async def query_stream(
    request: Request,
    user_id: str = Query(..., description="Owning user identifier"),
    query: str = Query(..., min_length=1, description="Natural-language question"),
    filters: Optional[str] = Query(
        default=None,
        description='JSON-encoded QueryFilters, e.g. {"doc_type":"invoice","session_id":"..."}',
    ),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Hybrid retrieval + Gemini streaming answer (SSE).

    Args:
        request:  FastAPI request (carries app.state).
        user_id:  Tenant filter for retrieval.
        query:    User question string.
        filters:  Optional JSON-encoded ``QueryFilters``.
        settings: Application settings.

    Returns:
        ``StreamingResponse`` with ``media_type="text/event-stream"``.
    """
    parsed_filters = QueryFilters()
    if filters:
        try:
            parsed_filters = QueryFilters.model_validate(json.loads(filters))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid filters JSON: {exc}",
            )

    retrieval_svc, answer_svc = _build_services(request, settings)
    doc_type = parsed_filters.doc_type.value if parsed_filters.doc_type else None

    try:
        ranked_chunks = await retrieval_svc.retrieve(
            query=query,
            user_id=user_id,
            doc_type=doc_type,
            session_id=parsed_filters.session_id,
        )
    except Exception as exc:
        await logger.aerror("query_stream.retrieval_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {exc}",
        )

    return StreamingResponse(
        answer_svc.answer_stream(query, ranked_chunks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
        },
    )
