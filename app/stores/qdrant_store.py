"""
app/stores/qdrant_store.py
==========================
Qdrant vector store wrapper.

Supports both local persistent mode (when ``qdrant_url`` is empty) and
cloud mode (when ``qdrant_url`` is set) with zero code change — the switch
is driven entirely by settings.

Collection initialisation is idempotent: calling ``initialize()`` on a
collection that already exists is a no-op.
"""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.models.domain import EmbeddedChunk, ScoredChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantStore:
    """
    Async-compatible wrapper around the Qdrant Python client.

    The qdrant-client is synchronous internally but is used inside async
    routes via FastAPI's thread-pool executor pattern (``run_in_executor``).
    For simplicity, all public methods here are synchronous — the calling
    services wrap them with ``asyncio.to_thread`` where necessary.

    Args:
        qdrant_url:       Cloud Qdrant URL, or empty string for local mode.
        qdrant_api_key:   API key for cloud mode.
        local_path:       Filesystem path used in local persistent mode.
        collection_name:  Qdrant collection to use.
        vector_size:      Embedding dimension (must match the model).
    """

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        local_path: str,
        collection_name: str,
        vector_size: int,
        batch_size: int = 32,
    ) -> None:
        self._qdrant_url = qdrant_url.strip()
        self._qdrant_api_key = qdrant_api_key
        self._local_path = local_path
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._batch_size = batch_size
        self._client: Optional[QdrantClient] = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Instantiate the Qdrant client.

        Uses local persistent mode when ``qdrant_url`` is empty, cloud mode
        otherwise.  Must be called once during app lifespan startup.
        """
        if self._qdrant_url:
            self._client = QdrantClient(
                url=self._qdrant_url,
                api_key=self._qdrant_api_key or None,
            )
            logger.info("qdrant_store.connected_cloud", url=self._qdrant_url)
        else:
            self._client = QdrantClient(path=self._local_path)
            logger.info("qdrant_store.connected_local", path=self._local_path)

    def close(self) -> None:
        """Close the Qdrant client."""
        if self._client:
            self._client.close()
            logger.info("qdrant_store.closed")

    def ping(self) -> bool:
        """
        Check Qdrant connectivity by listing collections.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("qdrant_store.ping_failed", error=str(exc))
            return False

    # ── Collection initialisation ─────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Idempotently create the collection and payload indexes.

        Safe to call on every startup — existing collections and indexes are
        left untouched.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info(
                "qdrant_store.collection_created",
                collection=self._collection_name,
                vector_size=self._vector_size,
            )
        else:
            logger.info(
                "qdrant_store.collection_exists", collection=self._collection_name
            )

        # Payload indexes for fast filtered search
        index_fields: dict[str, PayloadSchemaType] = {
            "user_id": PayloadSchemaType.KEYWORD,
            "session_id": PayloadSchemaType.KEYWORD,
            "doc_type": PayloadSchemaType.KEYWORD,
            "page_num": PayloadSchemaType.INTEGER,
            "role": PayloadSchemaType.KEYWORD,
        }
        for field, schema_type in index_fields.items():
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception:
                # Index already exists — qdrant-client raises on duplicate
                pass

        logger.info("qdrant_store.indexes_ready", collection=self._collection_name)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_chunks(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Batch-upsert a list of embedded chunks into the collection.

        Each point's ID is the ``chunk_id``.  The full chunk payload is stored
        alongside the vector for retrieval without a second lookup.  Chunks are
        split into batches of ``batch_size`` to avoid oversized requests.

        Args:
            embedded_chunks: List of ``EmbeddedChunk`` domain objects.
        """
        points: list[PointStruct] = []
        for ec in embedded_chunks:
            c = ec.chunk
            payload = {
                "user_id": c.user_id,
                "session_id": c.session_id,
                "doc_type": c.doc_type.value if c.doc_type else None,
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "role": c.role.value,
                "parent_chunk_id": c.parent_chunk_id,
                "entity_map": c.entity_map.model_dump(),
                "chunk_text": c.text,
                "source_drive_path": c.source_drive_path,
            }
            points.append(
                PointStruct(id=c.chunk_id, vector=ec.vector, payload=payload)
            )

        total = len(points)
        if total == 0:
            logger.warning("qdrant_store.upsert_empty_list")
            return

        try:
            for batch_start in range(0, total, self._batch_size):
                batch = points[batch_start : batch_start + self._batch_size]
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                )
        except Exception as exc:
            logger.error("qdrant_store.upsert_failed", error=str(exc))
            raise

        logger.info(
            "qdrant_store.upserted",
            count=total,
            collection=self._collection_name,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        top_k: int,
        doc_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[ScoredChunk]:
        """
        Perform a filtered vector search.

        ``user_id`` is always applied as a strict filter for tenant isolation.
        ``doc_type`` and ``session_id`` are optional additional filters.

        Args:
            query_vector: Query embedding (768 dims).
            user_id:      Mandatory tenant filter.
            top_k:        Number of results to return.
            doc_type:     Optional document type filter.
            session_id:   Optional session filter.

        Returns:
            List of ``ScoredChunk`` ordered by descending cosine similarity.
        """
        must_conditions: list[FieldCondition] = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        ]
        if doc_type:
            must_conditions.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )
        if session_id:
            must_conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )

        try:
            results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector,
                query_filter=Filter(must=must_conditions),
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("qdrant_store.search_failed", error=str(exc))
            raise

        return [
            ScoredChunk(
                chunk_id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    # ── Point fetch ───────────────────────────────────────────────────────────

    def get_by_ids(self, chunk_ids: list[str]) -> list[ScoredChunk]:
        """
        Fetch points by their IDs without a vector search.

        Used to retrieve parent page chunks after a child-chunk vector search.

        Args:
            chunk_ids: List of chunk UUID strings.

        Returns:
            List of ``ScoredChunk`` (score=0.0) for found IDs.
        """
        if not chunk_ids:
            return []

        records = self._client.retrieve(
            collection_name=self._collection_name,
            ids=chunk_ids,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk_id=str(r.id), score=0.0, payload=r.payload or {})
            for r in records
        ]
