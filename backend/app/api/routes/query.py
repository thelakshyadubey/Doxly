"""
app/api/routes/query.py
========================
QnA endpoint — non-streaming.

POST /query
    Body: {user_id, query, filters?}
    Returns: AnswerResult (full answer + citations + confidence)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from backend.app.config.settings import Settings, get_settings
from backend.app.models.api import GraphEdge, GraphNode, GraphResponse, QueryRequest, QueryResponse
from backend.app.services.answer_service import AnswerService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.stores.neo4j_store import Neo4jStore
from backend.app.stores.qdrant_store import QdrantStore
from backend.app.utils.logger import get_logger

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


def _build_services(
    request: Request, settings: Settings
) -> tuple[RetrievalService, AnswerService]:
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


@router.post(
    "",
    response_model=QueryResponse,
    summary="Document QnA",
    description="Retrieve relevant chunks via hybrid search and generate an answer using Gemini.",
)
async def query(
    body: QueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    retrieval_svc, answer_svc = _build_services(request, settings)

    session_id = body.filters.session_id

    try:
        ranked_chunks = await retrieval_svc.retrieve(
            query=body.query,
            user_id=body.user_id,
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


@router.get(
    "/graph",
    response_model=GraphResponse,
    summary="Reasoning subgraph for cited chunks",
    description=(
        "Return the Neo4j subgraph (Session → Chunk → Entity) for the given "
        "chunk IDs. Pass the chunk_ids from a /query citation list to visualise "
        "how the answer was derived."
    ),
)
async def query_graph(
    request: Request,
    user_id: str = Query(..., description="Owning user identifier"),
    chunk_ids: list[str] = Query(..., description="Chunk UUIDs from citation list"),
) -> GraphResponse:
    neo4j: Neo4jStore = request.app.state.neo4j_store
    try:
        raw = await neo4j.get_chunk_subgraph(chunk_ids, user_id)
    except Exception as exc:
        await logger.aerror("query.graph_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph query failed: {exc}",
        )
    return GraphResponse(
        nodes=[GraphNode(**n) for n in raw["nodes"]],
        edges=[GraphEdge(**e) for e in raw["edges"]],
    )
