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
from typing import AsyncIterator

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import auth, health, ingest, query
from app.config.settings import get_settings
from app.models.api import ErrorResponse
from app.services.drive_service import DriveService
from app.services.ocr_service import OCRService
from app.stores.neo4j_store import Neo4jStore
from app.stores.qdrant_store import QdrantStore
from app.stores.redis_store import RedisStore
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


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

    # 3. Qdrant
    qdrant_store = QdrantStore(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        local_path=settings.qdrant_local_path,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
        batch_size=settings.embedding_batch_size,
    )
    await asyncio.to_thread(qdrant_store.connect)
    await asyncio.to_thread(qdrant_store.initialize)
    app.state.qdrant_store = qdrant_store

    # 4. Neo4j
    neo4j_store = Neo4jStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )
    await neo4j_store.connect()
    await neo4j_store.initialize()
    app.state.neo4j_store = neo4j_store

    # 5. Redis
    redis_store = RedisStore(
        redis_url=settings.redis_url,
        session_threshold_seconds=settings.session_threshold_seconds,
    )
    await redis_store.connect()
    app.state.redis_store = redis_store

    # 6. OCR service (connect() creates the Vision client synchronously)
    ocr_service = OCRService(project_id=settings.google_cloud_project)
    await asyncio.to_thread(ocr_service.connect)
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
    await asyncio.to_thread(qdrant_store.close)
    await neo4j_store.close()
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

    # CORS — restrict in production by setting allowed origins in settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
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
