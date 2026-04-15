"""
app/api/routes/auth.py
=======================
Google OAuth 2.0 endpoints.

GET /auth/login?user_id={user_id}
    Redirects the user to Google's OAuth consent screen.
    Stores a CSRF ``state`` token in Redis bound to ``user_id``.

GET /auth/callback?code={code}&state={state}
    Handles Google's redirect after the user grants consent.
    Validates ``state``, exchanges ``code`` for tokens, and persists the
    tokens in Redis under ``tokens:{user_id}``.
    Returns a JSON confirmation — in production, redirect to your frontend.

GET /auth/status?user_id={user_id}
    Returns whether the user has completed OAuth authorization.

DELETE /auth/revoke?user_id={user_id}
    Removes the stored tokens from Redis (de-authorises the user).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse

from backend.app.config.settings import Settings, get_settings
from backend.app.services.auth_service import AuthService
from backend.app.stores.redis_store import RedisStore
from backend.app.utils.logger import get_logger

router = APIRouter(prefix="/auth", tags=["auth"])
logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_auth_service(request: Request, settings: Settings) -> AuthService:
    """Compose ``AuthService`` from app.state singletons."""
    redis: RedisStore = request.app.state.redis_store
    return AuthService(
        client_config=settings.oauth_client_config,
        redirect_uri=settings.google_oauth_redirect_uri,
        redis_store=redis,
    )


# ── GET /auth/login ───────────────────────────────────────────────────────────


@router.get(
    "/login",
    summary="Initiate Google OAuth 2.0 flow",
    description=(
        "Generates a Google consent-screen URL for the given user and redirects "
        "to it. The ``state`` parameter is stored in Redis for CSRF verification."
    ),
)
async def login(
    request: Request,
    user_id: str = Query(..., description="Opaque user identifier"),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    """
    Redirect the user to Google's OAuth consent screen.

    Args:
        request:  FastAPI request (carries app.state).
        user_id:  Identifier of the user initiating authorization.
        settings: Application settings.

    Returns:
        302 redirect to Google's authorization URL.
    """
    auth_svc = _build_auth_service(request, settings)
    auth_url = await auth_svc.get_authorization_url(user_id)
    await logger.ainfo("auth.login_redirected", user_id=user_id)
    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


# ── GET /auth/callback ────────────────────────────────────────────────────────


@router.get(
    "/callback",
    summary="OAuth 2.0 callback — exchange code for tokens",
    description=(
        "Receives Google's redirect after user grants consent. "
        "Validates ``state``, exchanges ``code`` for tokens, and stores them in Redis."
    ),
)
async def callback(
    request: Request,
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="CSRF state token from Google"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Handle the OAuth 2.0 callback from Google.

    Args:
        request:  FastAPI request (carries app.state).
        code:     Authorization code returned by Google.
        state:    State parameter returned by Google (must match Redis entry).
        settings: Application settings.

    Returns:
        JSON confirmation with ``user_id`` and ``authorized`` flag.

    Raises:
        422 if ``state`` is invalid or expired.
    """
    auth_svc = _build_auth_service(request, settings)
    redis: RedisStore = request.app.state.redis_store

    try:
        user_id, tokens = await auth_svc.exchange_code(code, state)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    await redis.save_user_tokens(user_id, tokens)
    await logger.ainfo("auth.callback_success", user_id=user_id)

    return {
        "authorized": True,
        "user_id": user_id,
        "message": (
            "Google Drive authorization complete. "
            "You may now upload documents via POST /ingest/upload."
        ),
    }


# ── GET /auth/status ──────────────────────────────────────────────────────────


@router.get(
    "/status",
    summary="Check user authorization status",
    description="Returns whether the given user has completed OAuth authorization.",
)
async def auth_status(
    request: Request,
    user_id: str = Query(..., description="Opaque user identifier"),
) -> dict:
    """
    Check whether a user has authorized Drive access.

    Args:
        request: FastAPI request (carries app.state).
        user_id: User to check.

    Returns:
        JSON with ``authorized`` boolean.
    """
    redis: RedisStore = request.app.state.redis_store
    authorized = await redis.has_user_tokens(user_id)
    return {"user_id": user_id, "authorized": authorized}


# ── DELETE /auth/revoke ───────────────────────────────────────────────────────


@router.delete(
    "/revoke",
    summary="Revoke Drive authorization",
    description="Removes the stored OAuth tokens for the given user from Redis.",
)
async def revoke(
    request: Request,
    user_id: str = Query(..., description="Opaque user identifier"),
) -> dict:
    """
    De-authorize a user by deleting their stored tokens.

    Args:
        request: FastAPI request (carries app.state).
        user_id: User whose tokens should be removed.

    Returns:
        JSON confirmation.
    """
    redis: RedisStore = request.app.state.redis_store
    await redis.delete_user_tokens(user_id)
    await logger.ainfo("auth.revoked", user_id=user_id)
    return {"user_id": user_id, "authorized": False, "message": "Tokens revoked."}
