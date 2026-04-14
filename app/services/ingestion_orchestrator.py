"""
app/services/ingestion_orchestrator.py
=======================================
Coordinates the full post-upload ingestion pipeline.

Pipeline triggered by POST /ingest/flush/{session_id}:

    1. Fetch session text from Redis
    2. Classify document type (Gemini)
    3. Create Drive folder hierarchy (ROOT/user/session/doc_type/)
    4. Upload raw OCR text to Drive
    5. Run coreference resolution (Gemini 2-pass)
    6. Upload resolved text to Drive
    7. Chunk + embed resolved text (tiktoken + Gemini embeddings)
    8. Upload chunk manifest JSON to Drive
    9. Upsert embedded chunks into Qdrant
   10. Write session + chunk + entity graph into Neo4j
   11. Mark session as INDEXED in Redis

Any unhandled exception transitions the session to FAILED and re-raises.
"""

from __future__ import annotations

import asyncio

from app.models.domain import DocType, SessionMetadata
from app.services.chunking_service import ChunkingService
from app.services.classification_service import ClassificationService
from app.services.coref_service import CorefService
from app.services.drive_service import UserDriveClient
from app.services.session_service import SessionService
from app.stores.neo4j_store import Neo4jStore
from app.stores.qdrant_store import QdrantStore
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionOrchestrator:
    """
    Coordinates all services involved in turning a closed session into indexed
    searchable chunks.

    Args:
        session_service:         Manages session lifecycle in Redis.
        classification_service:  Classifies document type via Gemini.
        coref_service:           Two-pass coreference resolution via Gemini.
        chunking_service:        Parent-child chunking and Gemini embedding.
        drive_client:            Per-user Drive client; uploads artefacts to the
                                 user's own Google Drive.
        qdrant_store:            Vector upsert target.
        neo4j_store:             Knowledge graph write target.
    """

    def __init__(
        self,
        session_service: SessionService,
        classification_service: ClassificationService,
        coref_service: CorefService,
        chunking_service: ChunkingService,
        drive_client: UserDriveClient,
        qdrant_store: QdrantStore,
        neo4j_store: Neo4jStore,
    ) -> None:
        self._sessions = session_service
        self._classifier = classification_service
        self._coref = coref_service
        self._chunking = chunking_service
        self._drive = drive_client
        self._qdrant = qdrant_store
        self._neo4j = neo4j_store

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(
        self, user_id: str, session_id: str
    ) -> dict:
        """
        Execute the full ingestion pipeline for a session.

        Args:
            user_id:    Owning user identifier.
            session_id: Session UUID to process.

        Returns:
            Dict with keys ``chunk_count``, ``entity_count``, ``doc_type``.

        Raises:
            ValueError:  If the session is not found in Redis.
            RuntimeError: If any pipeline step fails (session marked FAILED).
        """
        await self._sessions.mark_processing(user_id, session_id)

        try:
            result = await self._execute(user_id, session_id)
        except Exception as exc:
            await logger.aerror(
                "ingestion_orchestrator.pipeline_failed",
                session_id=session_id,
                error=str(exc),
            )
            await self._sessions.mark_failed(user_id, session_id)
            raise

        await self._sessions.mark_indexed(user_id, session_id)
        await logger.ainfo(
            "ingestion_orchestrator.pipeline_complete",
            session_id=session_id,
            **result,
        )
        return result

    # ── Pipeline steps ────────────────────────────────────────────────────────

    async def _execute(self, user_id: str, session_id: str) -> dict:
        """Internal pipeline — all steps in order."""

        # 1. Fetch session
        session: SessionMetadata | None = await self._sessions.get_session(user_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")

        raw_text = session.accumulated_text
        if not raw_text.strip():
            raise ValueError(f"Session {session_id} has no accumulated OCR text")

        # 2. Classify
        classification = await self._classifier.classify(raw_text)
        doc_type: DocType = classification.doc_type
        await logger.ainfo(
            "ingestion_orchestrator.classified",
            session_id=session_id,
            doc_type=doc_type,
            confidence=classification.confidence,
        )

        # 3. Create Drive folder: My Drive/{app_folder}/{session_id}/{doc_type}/
        drive_folder_id = await self._drive.create_session_folder(
            session_id, doc_type.value
        )
        await self._sessions.set_drive_folder(user_id, session_id, drive_folder_id)
        drive_folder_path = f"{session_id}/{doc_type.value}"

        # 4. Upload raw OCR text
        await self._drive.upload_text(
            drive_folder_id,
            f"ocr_raw_{session_id}.txt",
            raw_text,
        )

        # 5. Coreference resolution
        resolved_text, entity_map = await self._coref.resolve(raw_text)

        # 6. Upload resolved text
        await self._drive.upload_text(
            drive_folder_id,
            f"ocr_resolved_{session_id}.txt",
            resolved_text,
        )

        # 7. Chunk + embed
        embedded_chunks = await self._chunking.chunk_and_embed(
            resolved_text=resolved_text,
            entity_map=entity_map,
            session_id=session_id,
            user_id=user_id,
            doc_type=doc_type,
            source_drive_path=drive_folder_path,
        )

        # 8. Upload chunk manifest
        manifest = [
            {
                "chunk_id": ec.chunk.chunk_id,
                "page_num": ec.chunk.page_num,
                "role": ec.chunk.role.value,
                "chunk_index": ec.chunk.chunk_index,
                "text_preview": ec.chunk.text[:100],
            }
            for ec in embedded_chunks
        ]
        await self._drive.upload_json(
            drive_folder_id,
            f"chunk_manifest_{session_id}.json",
            manifest,
        )

        # 9. Upsert into Qdrant (sync client wrapped in thread)
        await asyncio.to_thread(self._qdrant.upsert_chunks, embedded_chunks)

        # 10. Write graph into Neo4j
        await self._neo4j.ensure_user(user_id)
        await self._neo4j.create_session_node(
            user_id=user_id,
            session_id=session_id,
            doc_type=doc_type.value,
            drive_folder_path=drive_folder_path,
        )

        child_chunks = [ec.chunk for ec in embedded_chunks if ec.chunk.role.value == "child"]
        for chunk in child_chunks:
            await self._neo4j.upsert_chunk(chunk)
            await self._neo4j.upsert_entities(chunk.chunk_id, entity_map)

        entity_count = len(entity_map.entities)
        chunk_count = len(child_chunks)

        return {
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "doc_type": doc_type.value,
        }
