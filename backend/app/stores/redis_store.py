"""
app/stores/redis_store.py
=========================
Async Redis client wrapper for session metadata and OAuth token persistence.

Uses redis-py's native asyncio support (``redis.asyncio``).

Key namespaces
--------------
session:{user_id}:{session_id}  — SessionMetadata, TTL = SESSION_THRESHOLD_SECONDS
tokens:{user_id}                — OAuth token dict, no TTL (refresh tokens are long-lived)
oauth_state:{state}             — state → user_id mapping for CSRF, TTL = 600s
"""

from __future__ import annotations

import json
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis

from backend.app.models.domain import SessionMetadata, SessionStatus
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

_SESSION_KEY = "session:{user_id}:{session_id}"
_TOKENS_KEY = "tokens:{user_id}"
_OAUTH_STATE_KEY = "oauth_state:{state}"
_OAUTH_STATE_TTL = 600  # seconds — must complete OAuth flow within 10 minutes


class RedisStore:
    """
    Thin async wrapper around Redis for session metadata and OAuth token operations.

    Args:
        redis_url:                 Full Redis connection URL.
        session_threshold_seconds: TTL applied to every session key.
    """

    def __init__(self, redis_url: str, session_threshold_seconds: int) -> None:
        self._redis_url = redis_url
        self._ttl = session_threshold_seconds
        self._client: Optional[Redis] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the connection pool. Must be called during app lifespan startup."""
        self._client = aioredis.from_url(
            self._redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await logger.ainfo("redis_store.connected", url=self._redis_url.split("@")[-1])

    async def close(self) -> None:
        """Close the connection pool gracefully."""
        if self._client:
            await self._client.aclose()
            await logger.ainfo("redis_store.closed")

    async def ping(self) -> bool:
        """
        Check connectivity.

        Returns:
            True if the server responds to PING, False otherwise.
        """
        try:
            return bool(await self._client.ping())
        except Exception as exc:
            await logger.awarning("redis_store.ping_failed", error=str(exc))
            return False

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def _session_key(self, user_id: str, session_id: str) -> str:
        return _SESSION_KEY.format(user_id=user_id, session_id=session_id)

    async def save_session(self, session: SessionMetadata) -> None:
        """
        Persist (create or overwrite) session metadata with TTL.

        Args:
            session: ``SessionMetadata`` domain object to persist.
        """
        key = self._session_key(session.user_id, session.session_id)
        await self._client.set(key, session.model_dump_json(), ex=self._ttl)
        await logger.adebug(
            "redis_store.session_saved",
            session_id=session.session_id,
            status=session.status,
        )

    async def get_session(self, user_id: str, session_id: str) -> Optional[SessionMetadata]:
        """
        Fetch session metadata by user and session ID.

        Returns:
            ``SessionMetadata`` if the key exists and has not expired, else ``None``.
        """
        raw = await self._client.get(self._session_key(user_id, session_id))
        if raw is None:
            return None
        return SessionMetadata.model_validate_json(raw)

    async def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete session metadata immediately."""
        await self._client.delete(self._session_key(user_id, session_id))
        await logger.adebug("redis_store.session_deleted", session_id=session_id)

    async def append_page_text(
        self, user_id: str, session_id: str, page_text: str
    ) -> int:
        """
        Append OCR page text to the session's accumulated text and reset TTL.

        Args:
            user_id:    User identifier.
            session_id: Session UUID.
            page_text:  Raw OCR text for the new page.

        Returns:
            Updated total page count.

        Raises:
            ValueError: If the session does not exist in Redis.
        """
        session = await self.get_session(user_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")

        if session.accumulated_text:
            session.accumulated_text += f"\n[PAGE_BREAK]\n{page_text}"
        else:
            session.accumulated_text = page_text

        session.page_count += 1
        await self.save_session(session)
        return session.page_count

    async def update_session_status(
        self, user_id: str, session_id: str, status: SessionStatus
    ) -> None:
        """
        Update the status field of an existing session.

        Raises:
            ValueError: If the session does not exist.
        """
        session = await self.get_session(user_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found for user {user_id}")
        session.status = status
        await self.save_session(session)

    # ── OAuth token storage ───────────────────────────────────────────────────

    def _tokens_key(self, user_id: str) -> str:
        return _TOKENS_KEY.format(user_id=user_id)

    async def save_user_tokens(self, user_id: str, tokens: dict) -> None:
        """
        Persist a user's OAuth token dict.

        Tokens include ``access_token``, ``refresh_token``, ``token_uri``,
        ``client_id``, ``client_secret``, ``scopes``, and ``expiry``.
        No TTL is applied — refresh tokens are long-lived and updated on each use.

        Args:
            user_id: Opaque user identifier.
            tokens:  Token dict produced by ``AuthService.exchange_code`` or
                     ``AuthService.refresh_tokens``.
        """
        await self._client.set(self._tokens_key(user_id), json.dumps(tokens))
        await logger.adebug("redis_store.tokens_saved", user_id=user_id)

    async def get_user_tokens(self, user_id: str) -> Optional[dict]:
        """
        Retrieve the stored OAuth tokens for a user.

        Args:
            user_id: Opaque user identifier.

        Returns:
            Token dict if found, ``None`` if the user has not authorised yet.
        """
        raw = await self._client.get(self._tokens_key(user_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def delete_user_tokens(self, user_id: str) -> None:
        """
        Remove a user's stored tokens (effectively de-authorises the user).

        Args:
            user_id: Opaque user identifier.
        """
        await self._client.delete(self._tokens_key(user_id))
        await logger.ainfo("redis_store.tokens_deleted", user_id=user_id)

    async def has_user_tokens(self, user_id: str) -> bool:
        """
        Check whether a user has completed OAuth authorisation.

        Args:
            user_id: Opaque user identifier.

        Returns:
            True if tokens exist in Redis, False otherwise.
        """
        return bool(await self._client.exists(self._tokens_key(user_id)))

    # ── OAuth state (CSRF protection) ─────────────────────────────────────────

    def _state_key(self, state: str) -> str:
        return _OAUTH_STATE_KEY.format(state=state)

    async def save_oauth_state(self, state: str, user_id: str) -> None:
        """
        Store the OAuth ``state`` parameter mapping to ``user_id``.

        Used during the callback to verify the request originated from this app
        and to recover which user initiated the flow.

        Args:
            state:   Random opaque string generated in ``AuthService.get_authorization_url``.
            user_id: User who initiated the OAuth flow.
        """
        await self._client.set(
            self._state_key(state), user_id, ex=_OAUTH_STATE_TTL
        )

    async def consume_oauth_state(self, state: str) -> Optional[str]:
        """
        Atomically retrieve and delete the user_id bound to ``state``.

        Uses ``GETDEL`` (Redis ≥ 6.2) so that a given state can never be
        consumed twice, preventing replay attacks.

        Args:
            state: OAuth state parameter from the callback query string.

        Returns:
            ``user_id`` if the state is valid and unexpired, ``None`` otherwise.
        """
        key = self._state_key(state)
        # GETDEL atomically reads and removes the key in a single round-trip.
        user_id: Optional[str] = await self._client.getdel(key)
        return user_id
