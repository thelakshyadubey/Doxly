"""
app/services/ingestion_orchestrator.py
=======================================
Coordinates the full post-upload ingestion pipeline.

Pipeline triggered by POST /ingest/flush/{session_id}:

    1.  Fetch session text from Redis
    2.  Parallel: classify + entity extraction (Pass 1 only) + fetch folder registry
    2b. Normalize folder label against existing labels (Gemini semantic routing)
    3.  Create Drive folder: {app_folder}/{canonical_label}/{datetime}/
    3b. Move uploaded images from pending folder to datetime folder (parallel)
    4.  Upload raw OCR text to datetime folder
    5.  Parallel: Drive folder setup + chunk + embed (tiktoken + Gemini embeddings)
    6.  Upsert embedded chunks into Qdrant
    7.  Write session + chunk + entity graph into Neo4j (UNWIND batched)
    8.  Register canonical label in Redis for future normalization
    9.  Clean up pending Drive folder (best-effort)
   10.  Mark session as INDEXED in Redis

Any unhandled exception transitions the session to FAILED and re-raises.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from backend.app.models.domain import SessionMetadata
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
        coref_service:           Entity extraction (Pass 1 only — no text rewriting).
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
        """Internal pipeline — parallel where possible."""

        # 1. Fetch session
        session: SessionMetadata | None = await self._sessions.get_session(user_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")

        raw_text = session.accumulated_text
        if not raw_text.strip():
            raise ValueError(f"Session {session_id} has no accumulated OCR text")

        pending_folder_id = session.drive_folder_id
        uploaded_file_ids = list(session.uploaded_file_ids)

        # 2. Parallel: classify + entity extraction (Pass 1 only) + fetch folder registry.
        #    All three are independent — run concurrently to overlap Gemini latency.
        classification, entity_map, folder_registry = await asyncio.gather(
            self._classifier.classify(raw_text),
            self._coref.extract_entities(raw_text),
            self._redis.get_folder_registry(user_id),
        )

        raw_folder_label: str = classification.folder_label or "General"
        topic_summary: str = classification.topic_summary

        await logger.ainfo(
            "ingestion_orchestrator.classified",
            session_id=session_id,
            raw_folder_label=raw_folder_label,
            topic_summary=topic_summary[:120],
            confidence=classification.confidence,
        )

        # 3. Route to existing folder (Gemini-assisted semantic matching) or
        #    create a new one.  Gemini understands that "RAG" and "Retrieval
        #    Augmented Generation" are the same topic; cosine similarity on
        #    short label strings does not.
        canonical_label = await self._classifier.route_to_folder(
            topic_summary=topic_summary,
            raw_label=raw_folder_label,
            folder_registry=folder_registry,
        )

        await logger.ainfo(
            "ingestion_orchestrator.label_resolved",
            session_id=session_id,
            canonical_label=canonical_label,
        )

        # 4. Compute drive path upfront — both Drive setup and chunking need it.
        datetime_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%I-%M-%S-%p")
        drive_folder_path = f"{canonical_label}/{datetime_str}"

        # 5. Parallel: Drive folder setup AND chunk+embed.
        #    Drive I/O and Gemini embedding are independent — run concurrently.
        async def _drive_setup() -> str:
            _, dt_folder_id = await self._drive.create_doc_type_datetime_folder(
                canonical_label, datetime_str
            )
            await self._sessions.set_drive_folder(user_id, session_id, dt_folder_id)

            # Move all uploaded images in parallel instead of one by one
            if pending_folder_id and uploaded_file_ids:
                move_results = await asyncio.gather(
                    *[
                        self._drive.move_file(fid, dt_folder_id, pending_folder_id)
                        for fid in uploaded_file_ids
                    ],
                    return_exceptions=True,
                )
                for fid, result in zip(uploaded_file_ids, move_results):
                    if isinstance(result, Exception):
                        await logger.awarning(
                            "ingestion_orchestrator.file_move_failed",
                            file_id=fid,
                            error=str(result),
                        )

            await self._drive.upload_text(dt_folder_id, "raw_ocr.txt", raw_text)
            return dt_folder_id

        datetime_folder_id, embedded_chunks = await asyncio.gather(
            _drive_setup(),
            self._chunking.chunk_and_embed(
                resolved_text=raw_text,
                entity_map=entity_map,
                session_id=session_id,
                user_id=user_id,
                source_drive_path=drive_folder_path,
            ),
        )

        # 6. Upsert into Qdrant (sync client wrapped in thread)
        await asyncio.to_thread(self._qdrant.upsert_chunks, embedded_chunks)

        # 7. Neo4j graph writes
        await self._neo4j.ensure_user(user_id)
        await self._neo4j.create_session_node(
            user_id=user_id,
            session_id=session_id,
            drive_folder_path=drive_folder_path,
        )

        child_chunks = [ec.chunk for ec in embedded_chunks if ec.chunk.role.value == "child"]

        # Upsert all chunk nodes in one UNWIND query, then entity links in one batch.
        await self._neo4j.upsert_chunks(child_chunks)
        await self._neo4j.upsert_entities_batch(
            [(c.chunk_id, entity_map) for c in child_chunks]
        )

        # 8. Register label + clean up pending folder — both best-effort, run together.
        async def _cleanup_pending() -> None:
            if not pending_folder_id:
                return
            try:
                await self._drive.delete_folder(pending_folder_id)
            except Exception as exc:
                await logger.awarning(
                    "ingestion_orchestrator.pending_folder_cleanup_failed",
                    pending_folder_id=pending_folder_id,
                    error=str(exc),
                )

        await asyncio.gather(
            self._redis.add_folder_to_registry(user_id, canonical_label, topic_summary),
            _cleanup_pending(),
        )

        entity_count = len(entity_map.entities)
        chunk_count = len(child_chunks)

        return {
            "chunk_count": chunk_count,
            "entity_count": entity_count,
            "folder_label": canonical_label,
            "drive_folder_path": drive_folder_path,
        }
