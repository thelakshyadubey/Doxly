"""
app/services/auth_service.py
=============================
Google OAuth 2.0 flow management for per-user Drive authorization.

Flow:
    1. ``get_authorization_url(user_id)``
       — Generates the Google consent-screen URL + random ``state`` param.
       — Stores ``state → user_id`` in Redis (TTL = 600 s, CSRF protection).
       — Returns the URL to redirect the user to.

    2. ``exchange_code(code, state)``
       — Validates ``state`` by consuming it from Redis (read + delete).
       — Exchanges the authorization ``code`` for access + refresh tokens.
       — Returns ``(user_id, tokens_dict)``; caller stores tokens in Redis.

The token dict format matches what ``DriveService.for_user()`` expects:
    {
        "token":          str,           # access token
        "refresh_token":  str,
        "token_uri":      str,
        "client_id":      str,
        "client_secret":  str,
        "scopes":         list[str],
        "expiry":         str | None,    # ISO-8601 datetime string
    }
"""

from __future__ import annotations

import asyncio
import secrets
from typing import Optional

from google_auth_oauthlib.flow import Flow

from app.stores.redis_store import RedisStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]


class AuthService:
    """
    Manages the Google OAuth 2.0 authorization code flow.

    Args:
        client_config:   OAuth client config dict (matches downloaded JSON format).
                         Typically ``settings.oauth_client_config``.
        redirect_uri:    Callback URL registered in Google Cloud Console.
        redis_store:     Redis store used for CSRF state persistence.
    """

    def __init__(
        self,
        client_config: dict,
        redirect_uri: str,
        redis_store: RedisStore,
    ) -> None:
        self._client_config = client_config
        self._redirect_uri = redirect_uri
        self._redis = redis_store

    # ── Step 1: Generate authorization URL ───────────────────────────────────

    async def get_authorization_url(self, user_id: str) -> str:
        """
        Generate the Google OAuth consent-screen URL for ``user_id``.

        A random ``state`` token is created, stored in Redis (TTL=600 s) bound
        to ``user_id``, and embedded in the authorization URL.  The state value
        is verified in ``exchange_code`` to prevent CSRF attacks.

        Args:
            user_id: Opaque user identifier that initiated the auth flow.

        Returns:
            Authorization URL to redirect the user to.
        """
        state = secrets.token_urlsafe(32)

        flow = Flow.from_client_config(
            self._client_config,
            scopes=_DRIVE_SCOPES,
            redirect_uri=self._redirect_uri,
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",        # force refresh_token on every new grant
            state=state,
        )

        await self._redis.save_oauth_state(state, user_id)
        await logger.adebug(
            "auth_service.authorization_url_generated",
            user_id=user_id,
            state=state[:8] + "…",  # partial — avoid logging full state
        )
        return auth_url

    # ── Step 2: Exchange code for tokens ─────────────────────────────────────

    async def exchange_code(
        self, code: str, state: str
    ) -> tuple[str, dict]:
        """
        Validate ``state``, exchange ``code`` for tokens, and return them.

        The ``state`` is consumed from Redis (read + delete) so it cannot be
        replayed.

        Args:
            code:  Authorization code from Google's callback query string.
            state: State parameter from Google's callback query string.

        Returns:
            Tuple of ``(user_id, tokens_dict)``.

        Raises:
            ValueError: If ``state`` is invalid or expired (not in Redis).
        """
        user_id = await self._redis.consume_oauth_state(state)
        if user_id is None:
            raise ValueError(
                "Invalid or expired OAuth state parameter — "
                "possible CSRF attempt or the 10-minute window has elapsed."
            )

        flow = Flow.from_client_config(
            self._client_config,
            scopes=_DRIVE_SCOPES,
            redirect_uri=self._redirect_uri,
            state=state,
        )
        # fetch_token makes an outbound HTTP request — offload to thread pool
        await asyncio.to_thread(flow.fetch_token, code=code)

        tokens = _credentials_to_dict(flow.credentials)
        await logger.ainfo(
            "auth_service.tokens_exchanged",
            user_id=user_id,
            has_refresh_token=bool(tokens.get("refresh_token")),
        )
        return user_id, tokens


# ── Helpers ───────────────────────────────────────────────────────────────────


def _credentials_to_dict(credentials) -> dict:
    """
    Serialise a ``google.oauth2.credentials.Credentials`` object to a plain
    dict suitable for JSON storage in Redis.

    Args:
        credentials: ``google.oauth2.credentials.Credentials`` instance.

    Returns:
        Token dict with string-safe values.
    """
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or []),
        "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
    }
