"""
app/api/dependencies.py
=======================
FastAPI dependency-injection providers.

All store and service singletons are attached to ``app.state`` during the
lifespan startup hook in ``app/main.py``.  Each ``get_*`` function here
simply pulls the pre-initialised object off request state, making every
route handler independently testable by overriding these dependencies.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status

from backend.app.config.settings import Settings, get_settings
from backend.app.stores.neo4j_store import Neo4jStore
from backend.app.stores.qdrant_store import QdrantStore
from backend.app.stores.redis_store import RedisStore


# ── Settings ──────────────────────────────────────────────────────────────────


def get_settings_dep() -> Settings:
    """
    Return the application settings singleton.

    Provided as a FastAPI dependency so it can be overridden in tests.
    """
    return get_settings()


# ── Stores ────────────────────────────────────────────────────────────────────


def get_qdrant(request: Request) -> QdrantStore:
    """
    Retrieve the ``QdrantStore`` singleton from app state.

    The store is initialised (connected + collection bootstrapped) once
    during the lifespan startup in ``app/main.py``.

    Args:
        request: Injected by FastAPI.

    Returns:
        The application-wide ``QdrantStore`` instance.

    Raises:
        503 if Qdrant failed to connect at startup.
    """
    store = request.app.state.qdrant_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store is currently unavailable. Please try again later.",
        )
    return store


def get_neo4j(request: Request) -> Neo4jStore:
    """
    Retrieve the ``Neo4jStore`` singleton from app state.

    Args:
        request: Injected by FastAPI.

    Returns:
        The application-wide ``Neo4jStore`` instance.

    Raises:
        503 if Neo4j failed to connect at startup.
    """
    store = request.app.state.neo4j_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph store is currently unavailable. Please try again later.",
        )
    return store


def get_redis(request: Request) -> RedisStore:
    """
    Retrieve the ``RedisStore`` singleton from app state.

    Args:
        request: Injected by FastAPI.

    Returns:
        The application-wide ``RedisStore`` instance.

    Raises:
        503 if Redis failed to connect at startup.
    """
    store = request.app.state.redis_store
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session store is currently unavailable. Please try again later.",
        )
    return store


# ── Gemini client ──────────────────────────────────────────────────────────────


def get_gemini(request: Request) -> Any:
    """
    Retrieve the ``google.generativeai`` GenerativeModel client from app state.

    The client is configured in the lifespan startup with the API key from
    settings.  Services that need Gemini receive this via DI rather than
    importing ``google.generativeai`` directly.

    Args:
        request: Injected by FastAPI.

    Returns:
        The application-wide ``GenerativeModel`` instance for reasoning tasks.
    """
    return request.app.state.gemini_model


def get_gemini_embed(request: Request) -> str:
    """
    Retrieve the Gemini embedding model name from app state.

    Args:
        request: Injected by FastAPI.

    Returns:
        Embedding model name string (e.g. ``"models/text-embedding-004"``).
    """
    return request.app.state.gemini_embed_model
