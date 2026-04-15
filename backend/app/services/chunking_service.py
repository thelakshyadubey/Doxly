"""
app/services/chunking_service.py
=================================
Parent-child chunking + Gemini embedding.

Parent chunks:
    One per page (split on [PAGE_BREAK]).  Stored with role=PARENT.

Child chunks:
    Sliding-window sub-chunks of each parent, sized by tiktoken token count.
    Each child has an entity-context header prepended before embedding:
        "[ENTITIES: Acme Corp, John Smith]\nchild text..."

Embedding:
    Gemini ``text-embedding-004`` via ``google.generativeai.embed_content``.
    Batched according to ``EMBEDDING_BATCH_SIZE`` from settings.
"""

from __future__ import annotations

import asyncio
from typing import Any

import google.generativeai as genai

from backend.app.models.domain import (
    Chunk,
    ChunkRole,
    DocType,
    EmbeddedChunk,
    EntityMap,
)
from backend.app.utils.id_generator import make_chunk_id
from backend.app.utils.logger import get_logger
from backend.app.utils.token_counter import split_into_chunks

logger = get_logger(__name__)

_PAGE_BREAK = "[PAGE_BREAK]"


class ChunkingService:
    """
    Converts a coreference-resolved document into embedded chunks.

    Args:
        embedding_model_name: Gemini embedding model ID (e.g. ``"models/text-embedding-004"``).
        chunk_size_tokens:    Max tokens per child chunk.
        chunk_overlap_tokens: Token overlap between consecutive child chunks.
        embedding_batch_size: Max chunks sent per embedding API call.
    """

    def __init__(
        self,
        embedding_model_name: str,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        embedding_batch_size: int,
    ) -> None:
        self._embed_model = embedding_model_name
        self._chunk_size = chunk_size_tokens
        self._overlap = chunk_overlap_tokens
        self._batch_size = embedding_batch_size

    # ── Public API ────────────────────────────────────────────────────────────

    async def chunk_and_embed(
        self,
        resolved_text: str,
        entity_map: EntityMap,
        session_id: str,
        user_id: str,
        doc_type: DocType | None,
        source_drive_path: str,
    ) -> list[EmbeddedChunk]:
        """
        Split the resolved document text into parent + child chunks and embed them.

        Args:
            resolved_text:     Coreference-resolved full document text.
            entity_map:        Entity map from the coref service.
            session_id:        Session UUID.
            user_id:           Owning user identifier.
            doc_type:          Classified document type.
            source_drive_path: Drive path for source attribution in payloads.

        Returns:
            Flat list of ``EmbeddedChunk`` — parents first, then all children.
        """
        pages = self._split_pages(resolved_text)
        entity_header = self._build_entity_header(entity_map)

        all_chunks: list[Chunk] = []

        for page_num, page_text in enumerate(pages, start=1):
            parent_chunk_id = make_chunk_id(session_id, page_num, 0)
            parent = Chunk(
                chunk_id=parent_chunk_id,
                session_id=session_id,
                user_id=user_id,
                page_num=page_num,
                chunk_index=0,
                role=ChunkRole.PARENT,
                text=page_text,
                enriched_text=page_text,  # parents are not entity-header-enriched
                parent_chunk_id=None,
                doc_type=doc_type,
                entity_map=entity_map,
                source_drive_path=source_drive_path,
            )
            all_chunks.append(parent)

            child_texts = split_into_chunks(page_text, self._chunk_size, self._overlap)
            for idx, child_text in enumerate(child_texts, start=1):
                enriched = f"{entity_header}\n{child_text}" if entity_header else child_text
                child = Chunk(
                    chunk_id=make_chunk_id(session_id, page_num, idx),
                    session_id=session_id,
                    user_id=user_id,
                    page_num=page_num,
                    chunk_index=idx,
                    role=ChunkRole.CHILD,
                    text=child_text,
                    enriched_text=enriched,
                    parent_chunk_id=parent_chunk_id,
                    doc_type=doc_type,
                    entity_map=entity_map,
                    source_drive_path=source_drive_path,
                )
                all_chunks.append(child)

        await logger.ainfo(
            "chunking_service.chunks_created",
            total=len(all_chunks),
            pages=len(pages),
            session_id=session_id,
        )

        embedded = await self._embed_in_batches(all_chunks)
        return embedded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_pages(self, text: str) -> list[str]:
        """
        Split text on ``[PAGE_BREAK]`` markers into per-page strings.

        Empty pages are discarded.

        Args:
            text: Full document text with ``[PAGE_BREAK]`` between pages.

        Returns:
            Ordered list of non-empty page strings.
        """
        pages = [p.strip() for p in text.split(_PAGE_BREAK)]
        return [p for p in pages if p]

    def _build_entity_header(self, entity_map: EntityMap) -> str:
        """
        Build the ``[ENTITIES: ...]`` header prepended to child chunk text.

        Args:
            entity_map: Extracted entity map.

        Returns:
            Header string, or empty string if no entities.
        """
        names = entity_map.canonical_names
        if not names:
            return ""
        return f"[ENTITIES: {', '.join(names)}]"

    async def _embed_in_batches(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed all chunks in batches, returning ``EmbeddedChunk`` objects.

        Args:
            chunks: Flat list of ``Chunk`` objects to embed.

        Returns:
            Matching list of ``EmbeddedChunk`` with vectors populated.
        """
        embedded: list[EmbeddedChunk] = []
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            texts = [c.enriched_text for c in batch]
            vectors = await asyncio.to_thread(self._embed_batch, texts)
            for chunk, vector in zip(batch, vectors):
                embedded.append(EmbeddedChunk(chunk=chunk, vector=vector))

        await logger.ainfo(
            "chunking_service.embedding_complete",
            embedded=len(embedded),
        )
        return embedded

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Call the Gemini embedding API for a batch of texts.

        When ``content`` is a list, ``embed_content`` returns
        ``{"embedding": [vec1, vec2, ...]}``.  When it is a single string it
        returns ``{"embedding": vec}`` (a flat list of floats).  This helper
        normalises both cases so callers always get a list of vectors.

        Args:
            texts: List of strings to embed (max ``EMBEDDING_BATCH_SIZE``).

        Returns:
            List of 768-dim float vectors in the same order as *texts*.
        """
        if not texts:
            return []

        result = genai.embed_content(
            model=self._embed_model,
            content=texts,
            task_type="retrieval_document",
        )
        embedding = result["embedding"]
        # Normalise: single string → flat list; list of strings → list of lists
        if texts and not isinstance(embedding[0], list):
            # Single-item batch or API returned a flat vector — wrap it
            return [embedding]
        return embedding
