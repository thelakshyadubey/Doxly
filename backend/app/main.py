"""
app/main.py
===========
FastAPI application entry point.

Startup (lifespan):
    1. Configure structured JSON logging.
    2. Configure Gemini API key.
    3. Connect and initialise all stores (Qdrant, Neo4j, Redis).
    4. Instantiate service singletons and attach to app.state.

Shutdown (lifespan):
    Gracefully close all store connections.

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Coroutine

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import auth
from backend.app.api.routes import health, ingest, query
from backend.app.config.settings import get_settings
from backend.app.models.api import ErrorResponse
from backend.app.services.drive_service import DriveService
from backend.app.services.ocr_service import OCRService
from backend.app.stores.neo4j_store import Neo4jStore
from backend.app.stores.qdrant_store import QdrantStore
from backend.app.stores.redis_store import RedisStore
from backend.app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 3  # seconds between attempts


async def _connect_with_retry(
    name: str,
    fn: Callable[[], Coroutine[Any, Any, None]],
) -> bool:
    """
    Call an async initialisation coroutine up to ``_RETRY_ATTEMPTS`` times.

    Returns True on success, False if all attempts fail.  On failure the app
    continues in degraded mode — callers must set the store to None and let
    the dependency layer return 503s.
    """
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            await fn()
            await logger.ainfo(f"startup.{name}_ready", attempt=attempt)
            return True
        except Exception as exc:
            await logger.awarning(
                f"startup.{name}_attempt_failed",
                attempt=attempt,
                max_attempts=_RETRY_ATTEMPTS,
                error=str(exc),
            )
            if attempt < _RETRY_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY)

    await logger.aerror(
        f"startup.{name}_unavailable",
        detail=(
            f"All {_RETRY_ATTEMPTS} connection attempts failed. "
            f"Routes that depend on {name} will return 503 until it is reachable."
        ),
    )
    return False


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Manage the full application lifecycle.

    All startup initialisation runs before the ``yield``; teardown runs after.
    Any exception during startup propagates and prevents the server from starting.
    """
    settings = get_settings()

    # 1. Logging
    configure_logging(settings.log_level)
    await logger.ainfo("startup.logging_configured", log_level=settings.log_level)

    # 2. Gemini
    genai.configure(api_key=settings.gemini_api_key)
    gemini_model = genai.GenerativeModel(settings.gemini_reasoning_model)
    app.state.gemini_model = gemini_model
    app.state.gemini_embed_model = settings.gemini_embedding_model
    await logger.ainfo(
        "startup.gemini_configured",
        reasoning_model=settings.gemini_reasoning_model,
        embedding_model=settings.gemini_embedding_model,
    )

    # 3. Qdrant — connect + initialize with retry; degrade gracefully on failure
    qdrant_store = QdrantStore(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
        batch_size=settings.embedding_batch_size,
    )

    async def _init_qdrant() -> None:
        await asyncio.to_thread(qdrant_store.connect)
        await asyncio.to_thread(qdrant_store.initialize)

    app.state.qdrant_store = qdrant_store if await _connect_with_retry("qdrant", _init_qdrant) else None

    # 4. Neo4j — connect + initialize with retry; degrade gracefully on failure
    neo4j_store = Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    async def _init_neo4j() -> None:
        await neo4j_store.connect()
        await neo4j_store.initialize()

    app.state.neo4j_store = neo4j_store if await _connect_with_retry("neo4j", _init_neo4j) else None

    # 5. Redis — required for auth; retry with graceful degradation
    redis_store = RedisStore(
        redis_url=settings.redis_url,
        session_threshold_seconds=settings.session_threshold_seconds,
    )

    async def _init_redis() -> None:
        await redis_store.connect()
        # Eagerly verify the connection so failures surface here, not at request time
        if not await redis_store.ping():
            raise ConnectionError("Redis ping failed")

    app.state.redis_store = redis_store if await _connect_with_retry("redis", _init_redis) else None

    # 6. OCR service (uses the already-configured Gemini model)
    ocr_service = OCRService(model=gemini_model)
    ocr_service.connect()
    app.state.ocr_service = ocr_service

    # 7. Drive service (factory — no shared connection, per-user clients built on demand)
    drive_service = DriveService(
        client_id=settings.google_oauth_client_id,
        client_secret=settings.google_oauth_client_secret,
        app_folder_name=settings.google_drive_app_folder_name,
    )
    drive_service.connect()  # no-op: just logs; synchronous is fine here
    app.state.drive_service = drive_service

    await logger.ainfo("startup.complete")

    yield  # ← server is live

    # ── Shutdown ──────────────────────────────────────────────────────────────
    await logger.ainfo("shutdown.starting")
    if app.state.qdrant_store is not None:
        await asyncio.to_thread(qdrant_store.close)
    if app.state.neo4j_store is not None:
        await neo4j_store.close()
    if app.state.redis_store is not None:
        await redis_store.close()
    await logger.ainfo("shutdown.complete")


# ── FastAPI app ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI application.

    Returns:
        Fully configured ``FastAPI`` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Document Intelligence API",
        description=(
            "Production-grade OCR + RAG pipeline: upload document images, "
            "extract text, resolve coreferences, chunk, embed, and query with "
            "hybrid vector + graph retrieval."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow_origins=["*"] is incompatible with allow_credentials=True per
    # the CORS spec. Explicitly list allowed origins instead.
    # settings.frontend_url is included automatically; extend the list below for
    # additional environments (staging, production).
    allowed_origins = list(
        {
            settings.frontend_url,
            "http://localhost:5173",   # Vite dev server default
            "http://localhost:4173",   # Vite preview server default
            "http://127.0.0.1:5173",
            "http://127.0.0.1:4173",
        }
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(ingest.router)
    app.include_router(query.router)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for unhandled exceptions.

        Logs the error with full stack trace and returns a structured JSON
        error body instead of a bare 500.
        """
        logger.error(
            "unhandled_exception",
            path=str(request.url),
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                message="An unexpected error occurred.",
                detail=str(exc),
            ).model_dump(),
        )

    return app


app = create_app()
