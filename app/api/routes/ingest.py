"""
app/api/routes/ingest.py
=========================
Document upload and session flush endpoints.

POST /ingest/upload
    Accepts a multipart image + user_id.
    Runs OCR, accumulates page text in session, uploads original image to the
    user's own Google Drive.
    Returns immediately with session_id and current page_count.

POST /ingest/flush/{session_id}
    Manually closes the session window and triggers the full pipeline:
    classification → coref → chunking → Qdrant + Neo4j writes.
    Returns chunk/entity counts and doc_type when complete.

Both endpoints require the user to have completed OAuth authorization first
(GET /auth/login → GET /auth/callback).  A 401 is returned with an auth hint
if no tokens are found for the user.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status

from app.config.settings import Settings, get_settings
from app.models.api import FlushResponse, UploadResponse
from app.models.domain import DocType, SessionStatus
from app.services.chunking_service import ChunkingService
from app.services.classification_service import ClassificationService
from app.services.coref_service import CorefService
from app.services.drive_service import DriveService, UserDriveClient
from app.services.ingestion_orchestrator import IngestionOrchestrator
from app.services.ocr_service import OCRService
from app.services.session_service import SessionService
from app.stores.neo4j_store import Neo4jStore
from app.stores.qdrant_store import QdrantStore
from app.stores.redis_store import RedisStore
from app.utils.logger import get_logger

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _require_drive_client(
    user_id: str,
    request: Request,
    settings: Settings,
) -> UserDriveClient:
    """
    Load the user's OAuth tokens from Redis and return a ``UserDriveClient``.

    Raises:
        HTTPException 401: If the user has not yet authorized Drive access.
    """
    redis: RedisStore = request.app.state.redis_store
    tokens = await redis.get_user_tokens(user_id)
    if tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"User '{user_id}' has not authorized Google Drive access. "
                f"Complete OAuth at GET /auth/login?user_id={user_id}"
            ),
        )
    drive: DriveService = request.app.state.drive_service
    return drive.for_user(tokens)


def _build_orchestrator(
    request: Request,
    settings: Settings,
    drive_client: UserDriveClient,
) -> IngestionOrchestrator:
    """
    Compose an ``IngestionOrchestrator`` from app.state singletons and the
    per-user Drive client.
    """
    qdrant: QdrantStore = request.app.state.qdrant_store
    neo4j: Neo4jStore = request.app.state.neo4j_store
    redis: RedisStore = request.app.state.redis_store
    gemini_model = request.app.state.gemini_model
    embed_model: str = request.app.state.gemini_embed_model

    session_svc = SessionService(redis, settings.session_threshold_seconds)
    classifier = ClassificationService(gemini_model)
    coref = CorefService(gemini_model)
    chunking = ChunkingService(
        embedding_model_name=embed_model,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        embedding_batch_size=settings.embedding_batch_size,
    )

    return IngestionOrchestrator(
        session_service=session_svc,
        classification_service=classifier,
        coref_service=coref,
        chunking_service=chunking,
        drive_client=drive_client,
        qdrant_store=qdrant,
        neo4j_store=neo4j,
    )


# ── POST /ingest/upload ───────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document page image",
    description=(
        "Upload a single image page. The OCR result is accumulated in the session window. "
        "The full pipeline is triggered when the window expires or /flush is called. "
        "Requires prior OAuth authorization via GET /auth/login."
    ),
)
async def upload_page(
    request: Request,
    user_id: str = Form(..., description="Opaque user identifier"),
    file: UploadFile = File(..., description="Image file (JPEG, PNG, TIFF, or PDF page)"),
    settings: Settings = Depends(get_settings),
) -> UploadResponse:
    """
    OCR a single image page and accumulate it into the session window.

    Steps:
    1. Verify the user has authorized Drive (load tokens from Redis).
    2. Compute session_id from (user_id, current_time_bucket).
    3. Run OCR on the uploaded image bytes.
    4. Create Drive folder on first page (placeholder doc_type "pending").
    5. Upload the original image to the user's Drive.
    6. Append OCR text to the session's accumulated text in Redis.
    7. Persist any refreshed OAuth tokens back to Redis.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Verify OAuth and get per-user Drive client
    drive_client = await _require_drive_client(user_id, request, settings)

    ocr_service: OCRService = request.app.state.ocr_service
    redis: RedisStore = request.app.state.redis_store
    session_svc = SessionService(redis, settings.session_threshold_seconds)

    # Resolve or create session
    session = await session_svc.get_or_create_session(user_id)
    session_id = session.session_id

    # OCR
    try:
        ocr_result = await ocr_service.extract_text(
            image_bytes, mime_type=file.content_type or "image/jpeg"
        )
    except Exception as exc:
        await logger.aerror("ingest.ocr_failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OCR failed: {exc}",
        )

    # Create Drive folder on first page
    if not session.drive_folder_id:
        try:
            folder_id = await drive_client.create_session_folder(session_id, "pending")
            await session_svc.set_drive_folder(user_id, session_id, folder_id)
        except Exception as exc:
            await logger.awarning(
                "ingest.drive_folder_creation_failed",
                session_id=session_id,
                error=str(exc),
            )
            folder_id = ""
    else:
        folder_id = session.drive_folder_id

    # Upload original image to user's Drive (best-effort)
    if folder_id:
        try:
            page_count_so_far = session.page_count + 1
            await drive_client.upload_bytes(
                folder_id,
                f"original_page_{page_count_so_far}{_ext(file.filename)}",
                image_bytes,
                mime_type=file.content_type or "image/jpeg",
            )
        except Exception as exc:
            await logger.awarning(
                "ingest.drive_upload_failed",
                session_id=session_id,
                error=str(exc),
            )

    # Accumulate OCR text
    page_count = await session_svc.record_page(user_id, session_id, ocr_result.text)

    # Persist potentially-refreshed tokens back to Redis
    await redis.save_user_tokens(user_id, drive_client.get_current_tokens())

    await logger.ainfo(
        "ingest.page_uploaded",
        session_id=session_id,
        page_count=page_count,
        ocr_chars=len(ocr_result.text),
    )

    return UploadResponse(
        session_id=session_id,
        page_count=page_count,
        status=SessionStatus.QUEUED,
    )


# ── POST /ingest/flush/{session_id} ──────────────────────────────────────────


@router.post(
    "/flush/{session_id}",
    response_model=FlushResponse,
    summary="Trigger full ingestion pipeline for a session",
    description=(
        "Manually close the session window and run classification → coref → "
        "chunking → Qdrant/Neo4j writes. Idempotent if the session is already INDEXED. "
        "Requires prior OAuth authorization via GET /auth/login."
    ),
)
async def flush_session(
    session_id: str,
    request: Request,
    user_id: str = Query(..., description="Owning user identifier"),
    settings: Settings = Depends(get_settings),
) -> FlushResponse:
    """
    Close the session and run the full ingestion pipeline synchronously.

    Returns FlushResponse with chunk_count, entity_count, doc_type on success.
    """
    redis: RedisStore = request.app.state.redis_store
    session_svc = SessionService(redis, settings.session_threshold_seconds)

    session = await session_svc.get_session(user_id, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired.",
        )

    if session.status == SessionStatus.INDEXED:
        return FlushResponse(
            session_id=session_id,
            chunk_count=0,
            entity_count=0,
            doc_type=session.doc_type or DocType.OTHER,
            status=SessionStatus.INDEXED,
        )

    # Verify OAuth and get per-user Drive client
    drive_client = await _require_drive_client(user_id, request, settings)

    orchestrator = _build_orchestrator(request, settings, drive_client)

    try:
        result = await orchestrator.run(user_id, session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion pipeline failed: {exc}",
        )

    # Persist potentially-refreshed tokens back to Redis
    await redis.save_user_tokens(user_id, drive_client.get_current_tokens())

    return FlushResponse(
        session_id=session_id,
        chunk_count=result["chunk_count"],
        entity_count=result["entity_count"],
        doc_type=DocType(result["doc_type"]),
        status=SessionStatus.INDEXED,
    )


# ── Utilities ─────────────────────────────────────────────────────────────────


def _ext(filename: str | None) -> str:
    """Return the file extension including the dot, or '.jpg' as fallback."""
    if not filename or "." not in filename:
        return ".jpg"
    return "." + filename.rsplit(".", 1)[-1].lower()
