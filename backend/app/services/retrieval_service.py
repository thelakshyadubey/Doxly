"""
app/services/retrieval_service.py
===================================
Hybrid vector + graph retrieval with Reciprocal Rank Fusion (RRF) re-ranking.

Pipeline per query
------------------
1. Extract named entities from the query via Gemini.
2. In parallel (asyncio.gather):
   a. Embed query → Qdrant vector search filtered by user_id.
   b. For each query entity → Neo4j get_chunks_by_entity.
   c. For top vector hits → fetch their parent page chunks from Qdrant.
3. Combine all results, deduplicate by chunk_id, compute RRF scores.
4. Return top FINAL_TOP_K ``RankedChunk`` objects.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

import google.generativeai as genai

from backend.app.models.domain import RankedChunk, ScoredChunk
from backend.app.stores.neo4j_store import Neo4jStore
from backend.app.stores.qdrant_store import QdrantStore
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

_ENTITY_EXTRACTION_PROMPT = """Extract all named entities from the query below.
Return ONLY valid JSON with no markdown, no explanation:
{{"entities": ["<entity1>", "<entity2>"]}}

Query: {query}"""

_RRF_K = 60  # standard RRF constant


class RetrievalService:
    """
    Hybrid retrieval combining dense vector search (Qdrant) and
    entity-based graph traversal (Neo4j), fused via Reciprocal Rank Fusion.

    Args:
        qdrant_store:        Initialised ``QdrantStore`` instance.
        neo4j_store:         Initialised ``Neo4jStore`` instance.
        embedding_model:     Gemini embedding model name string.
        reasoning_model:     Configured ``GenerativeModel`` for entity extraction.
        retrieval_top_k:     Candidates fetched from each source.
        final_top_k:         Chunks returned after RRF re-ranking.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        neo4j_store: Neo4jStore,
        embedding_model: str,
        reasoning_model: Any,
        retrieval_top_k: int,
        final_top_k: int,
    ) -> None:
        self._qdrant = qdrant_store
        self._neo4j = neo4j_store
        self._embed_model = embedding_model
        self._model = reasoning_model
        self._top_k = retrieval_top_k
        self._final_k = final_top_k

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        user_id: str,
        doc_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[RankedChunk]:
        """
        Run the full hybrid retrieval pipeline for a query.

        Args:
            query:      Natural-language question string.
            user_id:    Mandatory tenant filter.
            doc_type:   Optional document type filter for Qdrant search.
            session_id: Optional session filter for Qdrant search.

        Returns:
            Ordered list of up to ``FINAL_TOP_K`` ``RankedChunk`` objects.
        """
        # Step 1: Extract query entities
        query_entities = await self._extract_query_entities(query)

        # Step 2: Parallel retrieval
        query_vector = await asyncio.to_thread(self._embed_query, query)

        vector_task = asyncio.to_thread(
            self._qdrant.search,
            query_vector,
            user_id,
            self._top_k,
            doc_type,
            session_id,
        )
        graph_task = self._graph_search(query_entities, user_id)

        vector_hits, graph_chunk_ids = await asyncio.gather(vector_task, graph_task)

        # Step 2c: Fetch parent chunks for top vector hits
        parent_ids = [
            h.payload.get("parent_chunk_id")
            for h in vector_hits
            if h.payload.get("parent_chunk_id")
        ]
        parent_hits: list[ScoredChunk] = await asyncio.to_thread(
            self._qdrant.get_by_ids, list(set(parent_ids))
        )

        # Step 3: Fetch graph-matched chunks from Qdrant
        graph_hits: list[ScoredChunk] = await asyncio.to_thread(
            self._qdrant.get_by_ids, list(set(graph_chunk_ids))
        )

        # Step 4: RRF fusion
        ranked = self._rrf_fuse(vector_hits, graph_hits, parent_hits)

        await logger.ainfo(
            "retrieval_service.retrieved",
            query_len=len(query),
            vector_hits=len(vector_hits),
            graph_hits=len(graph_hits),
            final=len(ranked),
        )
        return ranked

    # ── Entity extraction ─────────────────────────────────────────────────────

    async def _extract_query_entities(self, query: str) -> list[str]:
        """
        Extract named entities from the user query for graph lookup.

        Args:
            query: Natural-language query string.

        Returns:
            List of entity name strings (may be empty on failure).
        """
        prompt = _ENTITY_EXTRACTION_PROMPT.format(query=query)
        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            raw = re.sub(r"```(?:json)?|```", "", response.text.strip())
            data = json.loads(raw)
            entities = [str(e) for e in data.get("entities", []) if e]
            await logger.adebug(
                "retrieval_service.query_entities", entities=entities
            )
            return entities
        except Exception as exc:
            await logger.awarning(
                "retrieval_service.entity_extraction_failed", error=str(exc)
            )
            return []

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        """
        Embed the query string using Gemini text-embedding-004.

        Uses ``retrieval_query`` task type (vs. ``retrieval_document`` for chunks).

        Args:
            query: Query string.

        Returns:
            768-dim float vector.
        """
        result = genai.embed_content(
            model=self._embed_model,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]

    # ── Graph search ──────────────────────────────────────────────────────────

    async def _graph_search(
        self, entities: list[str], user_id: str
    ) -> list[str]:
        """
        Retrieve chunk IDs from Neo4j for all query entities.

        Args:
            entities: Entity names extracted from the query.
            user_id:  Tenant filter.

        Returns:
            Deduplicated list of chunk_id strings.
        """
        if not entities:
            return []

        tasks = [
            self._neo4j.get_chunks_by_entity(entity, user_id)
            for entity in entities
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_ids: set[str] = set()
        for result in results:
            if isinstance(result, list):
                chunk_ids.update(result)
            else:
                await logger.awarning(
                    "retrieval_service.graph_search_error", error=str(result)
                )
        return list(chunk_ids)

    # ── RRF fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        vector_hits: list[ScoredChunk],
        graph_hits: list[ScoredChunk],
        parent_hits: list[ScoredChunk],
    ) -> list[RankedChunk]:
        """
        Apply Reciprocal Rank Fusion across three ranked lists and return
        the top ``FINAL_TOP_K`` chunks.

        RRF score = Σ  1 / (k + rank_i)   for each source list i.

        Args:
            vector_hits: Ordered Qdrant results (by cosine similarity).
            graph_hits:  Qdrant points fetched via Neo4j entity lookup (unordered).
            parent_hits: Parent page chunks fetched for top vector results.

        Returns:
            Ordered list of up to ``FINAL_TOP_K`` ``RankedChunk`` objects.
        """
        rrf_scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        def _add_list(hits: list[ScoredChunk], weight: float = 1.0) -> None:
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight / (_RRF_K + rank)
                if cid not in payloads:
                    payloads[cid] = hit.payload

        _add_list(vector_hits, weight=1.0)
        _add_list(graph_hits, weight=1.0)
        _add_list(parent_hits, weight=0.5)  # parents get lower weight

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[: self._final_k]

        ranked: list[RankedChunk] = []
        for cid in top_ids:
            p = payloads[cid]
            ranked.append(
                RankedChunk(
                    chunk_id=cid,
                    rrf_score=rrf_scores[cid],
                    chunk_text=p.get("chunk_text", ""),
                    source_drive_path=p.get("source_drive_path", ""),
                    doc_type=p.get("doc_type", ""),
                    page_num=int(p.get("page_num", 0)),
                )
            )
        return ranked
