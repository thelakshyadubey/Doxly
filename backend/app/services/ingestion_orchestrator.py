"""
app/services/ingestion_orchestrator.py
=======================================
Coordinates the full post-upload ingestion pipeline.

Pipeline triggered by POST /ingest/flush/{session_id}:

    1.  Fetch session text from Redis
    2.  Classify document type + free-form folder label (Gemini)
    2b. Normalize folder label against existing labels (embedding cosine similarity)
    3.  Create Drive folder: {app_folder}/{canonical_label}/{datetime}/
    3b. Move uploaded images from pending folder to datetime folder
    4.  Upload raw OCR text to datetime folder
    5.  Run coreference resolution (Gemini 2-pass)
    6.  Chunk + embed resolved text (tiktoken + Gemini embeddings)
    7.  Upsert embedded chunks into Qdrant
    8.  Write session + chunk + entity graph into Neo4j
    9.  Register canonical label in Redis for future normalization
   10.  Clean up pending Drive folder (best-effort)
   11.  Mark session as INDEXED in Redis

Any unhandled exception transitions the session to FAILED and re-raises.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from backend.app.models.domain import DocType, SessionMetadata
from backend.app.services.chunking_service import ChunkingService
from backend.app.services.classification_service import ClassificationService
from backend.app.services.coref_service import CorefService
from backend.app.services.drive_service import UserDriveClient
from backend.app.services.session_service import SessionService
from backend.app.stores.neo4j_store import Neo4jStore
from backend.app.stores.qdrant_store import QdrantStore
from backend.app.stores.redis_store import RedisStore
from backend.app.utils.logger import get_logger

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
        redis_store:             Needed for folder-label registry reads/writes.
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
        redis_store: RedisStore,
    ) -> None:
        self._sessions = session_service
        self._classifier = classification_service
        self._coref = coref_service
        self._chunking = chunking_service
        self._drive = drive_client
        self._qdrant = qdrant_store
        self._neo4j = neo4j_store
        self._redis = redis_store

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

        pending_folder_id = session.drive_folder_id
        uploaded_file_ids = list(session.uploaded_file_ids)

        # 2. Classify → doc_type (enum for Qdrant/Neo4j) + folder_label (free-form for Drive)
        classification = await self._classifier.classify(raw_text)
        doc_type: DocType = classification.doc_type
        raw_folder_label: str = classification.folder_label or doc_type.value.replace("_", " ").title()

        await logger.ainfo(
            "ingestion_orchestrator.classified",
            session_id=session_id,
            doc_type=doc_type,
            raw_folder_label=raw_folder_label,
            confidence=classification.confidence,
        )

        # 2b. Normalize folder label against existing ones (embedding cosine similarity)
        existing_labels = await self._redis.get_folder_labels(user_id)
        canonical_label = await self._classifier.normalize_label(raw_folder_label, existing_labels)

        await logger.ainfo(
            "ingestion_orchestrator.label_resolved",
            session_id=session_id,
            canonical_label=canonical_label,
        )

        # 3. Create Drive folder: {app_folder}/{canonical_label}/{datetime}/
        datetime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        _, datetime_folder_id = await self._drive.create_doc_type_datetime_folder(
            canonical_label, datetime_str
        )
        drive_folder_path = f"{canonical_label}/{datetime_str}"
        await self._sessions.set_drive_folder(user_id, session_id, datetime_folder_id)

        # 3b. Move uploaded images from pending folder to the datetime folder
        if pending_folder_id and uploaded_file_ids:
            for file_id in uploaded_file_ids:
                try:
                    await self._drive.move_file(file_id, datetime_folder_id, pending_folder_id)
                except Exception as exc:
                    await logger.awarning(
                        "ingestion_orchestrator.file_move_failed",
                        file_id=file_id,
                        error=str(exc),
                    )

        # 4. Upload raw OCR text to the datetime folder
        await self._drive.upload_text(
            datetime_folder_id,
            "raw_ocr.txt",
            raw_text,
        )

        # 5. Coreference resolution
        resolved_text, entity_map = await self._coref.resolve(raw_text)

        # 6. Chunk + embed
        embedded_chunks = await self._chunking.chunk_and_embed(
            resolved_text=resolved_text,
            entity_map=entity_map,
            session_id=session_id,
            user_id=user_id,
            doc_type=doc_type,
            source_drive_path=drive_folder_path,
        )

        # 7. Upsert into Qdrant (sync client wrapped in thread)
        await asyncio.to_thread(self._qdrant.upsert_chunks, embedded_chunks)

        # 8. Write graph into Neo4j
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

        # 9. Register canonical label in Redis for future normalization
        await self._redis.add_folder_label(user_id, canonical_label)

        # 10. Clean up pending folder (best-effort — do not fail the pipeline)
        if pending_folder_id:
            try:
                await self._drive.delete_folder(pending_folder_id)
            except Exception as exc:
                await logger.awarning(
                    "ingestion_orchestrator.pending_folder_cleanup_failed",
                    pending_folder_id=pending_folder_id,
                    error=str(exc),
                )

        entity_count = len(entity_map.entities)
        chunk_count = len(child_chunks)

        return {
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "doc_type": doc_type.value,
            "folder_label": canonical_label,
            "drive_folder_path": drive_folder_path,
        }
