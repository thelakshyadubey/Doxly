"""
app/services/session_service.py
================================
Session windowing logic.

Computes session IDs deterministically from ``(user_id, timestamp_bucket)``
and manages the Redis-backed session lifecycle: creation, page accumulation,
and status transitions.

Session window rule:
    session_id = UUID5("session" | user_id | str(floor(ts / threshold)))
    All uploads within the same bucket share a session ID.
"""

from __future__ import annotations

import time

from backend.app.models.domain import SessionMetadata, SessionStatus
from backend.app.stores.redis_store import RedisStore
from backend.app.utils.id_generator import compute_timestamp_bucket, make_session_id
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionService:
    """
    Manages upload sessions using Redis-backed windowed state.

    Args:
        redis_store:               Injected ``RedisStore`` instance.
        session_threshold_seconds: Window duration in seconds (from settings).
    """

    def __init__(
        self,
        redis_store: RedisStore,
        session_threshold_seconds: int,
    ) -> None:
        self._store = redis_store
        self._threshold = session_threshold_seconds

    # ── Session resolution ────────────────────────────────────────────────────

    def compute_session_id(self, user_id: str, upload_time: float | None = None) -> str:
        """
        Compute the session ID for a given user at a given time.

        Args:
            user_id:     Opaque user identifier.
            upload_time: POSIX timestamp.  Defaults to ``time.time()`` if None.

        Returns:
            Deterministic session UUID string.
        """
        ts = upload_time if upload_time is not None else time.time()
        bucket = compute_timestamp_bucket(ts, self._threshold)
        return make_session_id(user_id, bucket)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def get_or_create_session(
        self,
        user_id: str,
        upload_time: float | None = None,
    ) -> SessionMetadata:
        """
        Return an existing open session for this user/window or create a new one.

        Args:
            user_id:     Opaque user identifier.
            upload_time: POSIX timestamp for bucket computation.

        Returns:
            ``SessionMetadata`` — either retrieved from Redis or freshly created.
        """
        ts = upload_time if upload_time is not None else time.time()
        bucket = compute_timestamp_bucket(ts, self._threshold)
        session_id = make_session_id(user_id, bucket)

        existing = await self._store.get_session(user_id, session_id)
        if existing is not None:
            await logger.adebug(
                "session_service.session_found",
                session_id=session_id,
                page_count=existing.page_count,
            )
            return existing

        # First upload in this window — create session
        session = SessionMetadata(
            user_id=user_id,
            session_id=session_id,
            timestamp_bucket=bucket,
            status=SessionStatus.OPEN,
        )
        await self._store.save_session(session)
        await logger.ainfo(
            "session_service.session_created",
            session_id=session_id,
            user_id=user_id,
            bucket=bucket,
        )
        return session

    async def record_page(
        self,
        user_id: str,
        session_id: str,
        page_text: str,
    ) -> int:
        """
        Append a new OCR page to the session's accumulated text.

        Args:
            user_id:    Owning user.
            session_id: Session UUID.
            page_text:  Raw OCR text for the new page.

        Returns:
            Updated total page count for the session.
        """
        count = await self._store.append_page_text(user_id, session_id, page_text)
        await logger.ainfo(
            "session_service.page_recorded",
            session_id=session_id,
            page_count=count,
        )
        return count

    async def set_drive_folder(
        self,
        user_id: str,
        session_id: str,
        drive_folder_id: str,
    ) -> None:
        """
        Persist the Drive folder ID into session metadata.

        Called after ``drive_service.create_session_folder``.

        Args:
            user_id:         Owning user.
            session_id:      Session UUID.
            drive_folder_id: Drive folder ID returned by DriveService.
        """
        session = await self._store.get_session(user_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        session.drive_folder_id = drive_folder_id
        await self._store.save_session(session)

    async def mark_queued(self, user_id: str, session_id: str) -> None:
        """
        Transition session status to QUEUED (window closed, pipeline pending).

        Args:
            user_id:    Owning user.
            session_id: Session UUID.
        """
        await self._store.update_session_status(user_id, session_id, SessionStatus.QUEUED)
        await logger.ainfo("session_service.status_queued", session_id=session_id)

    async def mark_processing(self, user_id: str, session_id: str) -> None:
        """
        Transition session status to PROCESSING (pipeline running).

        Args:
            user_id:    Owning user.
            session_id: Session UUID.
        """
        await self._store.update_session_status(
            user_id, session_id, SessionStatus.PROCESSING
        )

    async def mark_indexed(self, user_id: str, session_id: str) -> None:
        """
        Transition session status to INDEXED (fully ingested).

        Args:
            user_id:    Owning user.
            session_id: Session UUID.
        """
        await self._store.update_session_status(user_id, session_id, SessionStatus.INDEXED)
        await logger.ainfo("session_service.status_indexed", session_id=session_id)

    async def mark_failed(
        self, user_id: str, session_id: str
    ) -> None:
        """
        Transition session status to FAILED.

        Args:
            user_id:    Owning user.
            session_id: Session UUID.
        """
        await self._store.update_session_status(user_id, session_id, SessionStatus.FAILED)
        await logger.aerror("session_service.status_failed", session_id=session_id)

    async def get_session(self, user_id: str, session_id: str) -> SessionMetadata | None:
        """
        Retrieve session metadata.

        Args:
            user_id:    Owning user.
            session_id: Session UUID.

        Returns:
            ``SessionMetadata`` or ``None`` if not found / TTL expired.
        """
        return await self._store.get_session(user_id, session_id)
