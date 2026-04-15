"""
app/services/drive_service.py
==============================
Per-user Google Drive operations via OAuth 2.0.

``DriveService`` is a lightweight factory that holds the OAuth client config
(client_id, client_secret, app_folder_name).  Call ``for_user(tokens)`` to get
a ``UserDriveClient`` scoped to a single user's Drive.

``UserDriveClient`` wraps ``google.oauth2.credentials.Credentials`` built from
the token dict stored in Redis.  After any Drive operation that may refresh the
access token, call ``get_current_tokens()`` to retrieve the (possibly updated)
token dict and save it back to Redis.

Folder hierarchy created per session (in the *user's own* Drive):
    My Drive / {app_folder_name} / {session_id} / {doc_type} /
        original_page_{n}.jpg          — raw uploaded image bytes
        ocr_raw_{session_id}.txt       — concatenated raw OCR text
        ocr_resolved_{session_id}.txt  — coreference-resolved text
        chunk_manifest_{session_id}.json — serialised chunk list
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Optional

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
]


class UserDriveClient:
    """
    Drive v3 client bound to a single user's OAuth 2.0 credentials.

    Args:
        credentials:     ``google.oauth2.credentials.Credentials`` for this user.
        app_folder_name: Top-level folder name created in the user's own Drive.
    """

    def __init__(self, credentials: Credentials, app_folder_name: str) -> None:
        self._credentials = credentials
        self._app_folder_name = app_folder_name
        self._service = build(
            "drive", "v3", credentials=self._credentials, cache_discovery=False
        )

    # ── Token refresh access ──────────────────────────────────────────────────

    def get_current_tokens(self) -> dict:
        """
        Return the current token dict (access token may have been refreshed).

        Callers should save this back to Redis after each use so the updated
        access token is available for the next request.

        Returns:
            Token dict with keys: ``token``, ``refresh_token``, ``token_uri``,
            ``client_id``, ``client_secret``, ``scopes``, ``expiry``.
        """
        return {
            "token": self._credentials.token,
            "refresh_token": self._credentials.refresh_token,
            "token_uri": self._credentials.token_uri,
            "client_id": self._credentials.client_id,
            "client_secret": self._credentials.client_secret,
            "scopes": list(self._credentials.scopes or _DRIVE_SCOPES),
            "expiry": (
                self._credentials.expiry.isoformat()
                if self._credentials.expiry
                else None
            ),
        }

    # ── Folder management ─────────────────────────────────────────────────────

    async def get_or_create_folder(self, name: str, parent_id: str) -> str:
        """
        Return the Drive folder ID for *name* under *parent_id*, creating it
        if it does not already exist.

        Args:
            name:      Folder display name.
            parent_id: Parent folder Drive ID.

        Returns:
            Drive folder ID string.
        """
        return await asyncio.to_thread(self._sync_get_or_create_folder, name, parent_id)

    def _sync_get_or_create_folder(self, name: str, parent_id: str) -> str:
        # Escape single quotes in name to avoid query injection
        safe_name = name.replace("'", "\\'")
        query = (
            f"name='{safe_name}' and "
            f"mimeType='application/vnd.google-apps.folder' and "
            f"'{parent_id}' in parents and trashed=false"
        )
        response = (
            self._service.files()
            .list(q=query, fields="files(id)", spaces="drive")
            .execute()
        )
        files = response.get("files", [])
        if files:
            return files[0]["id"]

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self._service.files().create(body=metadata, fields="id").execute()
        logger.info("drive_service.folder_created", name=name, parent=parent_id)
        return folder["id"]

    async def _get_root_folder_id(self) -> str:
        """
        Resolve (or lazily create) the app root folder inside 'My Drive'.

        The app folder lives directly under the Drive root ('root' is the
        Drive API alias for 'My Drive').

        Returns:
            Drive folder ID of the app root folder.
        """
        return await self.get_or_create_folder(self._app_folder_name, "root")

    async def create_session_folder(self, session_id: str, doc_type: str) -> str:
        """
        Create the full folder path for a session and return the leaf folder ID.

        Path: My Drive / {app_folder_name} / {session_id} / {doc_type}

        Args:
            session_id: Session UUID string.
            doc_type:   Classified document type string.

        Returns:
            Drive folder ID of the leaf ``doc_type`` folder.
        """
        root_folder = await self._get_root_folder_id()
        session_folder = await self.get_or_create_folder(session_id, root_folder)
        doc_folder = await self.get_or_create_folder(doc_type, session_folder)
        return doc_folder

    # ── File uploads ──────────────────────────────────────────────────────────

    async def upload_bytes(
        self,
        folder_id: str,
        filename: str,
        content_bytes: bytes,
        mime_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload raw bytes as a new file in the given Drive folder.

        Args:
            folder_id:     Target Drive folder ID.
            filename:      Desired file name.
            content_bytes: File content.
            mime_type:     MIME type for the uploaded file.

        Returns:
            Drive file ID of the newly created file.
        """
        return await asyncio.to_thread(
            self._sync_upload_bytes, folder_id, filename, content_bytes, mime_type
        )

    def _sync_upload_bytes(
        self,
        folder_id: str,
        filename: str,
        content_bytes: bytes,
        mime_type: str,
    ) -> str:
        metadata = {"name": filename, "parents": [folder_id]}
        media = MediaIoBaseUpload(io.BytesIO(content_bytes), mimetype=mime_type)
        file = (
            self._service.files()
            .create(body=metadata, media_body=media, fields="id,webViewLink")
            .execute()
        )
        logger.info(
            "drive_service.file_uploaded",
            filename=filename,
            folder_id=folder_id,
            file_id=file["id"],
        )
        return file["id"]

    async def upload_text(self, folder_id: str, filename: str, text: str) -> str:
        """
        Upload a UTF-8 text string as a plain-text file.

        Args:
            folder_id: Target Drive folder ID.
            filename:  Desired file name.
            text:      Text content to write.

        Returns:
            Drive file ID.
        """
        return await self.upload_bytes(
            folder_id, filename, text.encode("utf-8"), mime_type="text/plain"
        )

    async def upload_json(
        self, folder_id: str, filename: str, data: dict | list
    ) -> str:
        """
        Serialise *data* to JSON and upload it to Drive.

        Args:
            folder_id: Target Drive folder ID.
            filename:  Desired file name (should end with ``.json``).
            data:      JSON-serialisable Python object.

        Returns:
            Drive file ID.
        """
        return await self.upload_bytes(
            folder_id,
            filename,
            json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
            mime_type="application/json",
        )

    async def get_file_web_link(self, file_id: str) -> str:
        """
        Return the browser-accessible webViewLink for a Drive file.

        Args:
            file_id: Drive file ID.

        Returns:
            URL string, or empty string if not available.
        """
        result = await asyncio.to_thread(
            lambda: self._service.files()
            .get(fileId=file_id, fields="webViewLink")
            .execute()
        )
        return result.get("webViewLink", "")


class DriveService:
    """
    Factory that creates per-user ``UserDriveClient`` instances.

    Holds the OAuth 2.0 client credentials and app folder name, which are
    shared across all users.  Individual user tokens are stored in Redis and
    passed in at request time.

    Args:
        client_id:       OAuth 2.0 client ID.
        client_secret:   OAuth 2.0 client secret.
        token_uri:       Google token endpoint URI.
        app_folder_name: Top-level folder name in each user's Drive.
    """

    _TOKEN_URI = "https://oauth2.googleapis.com/token"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        app_folder_name: str,
        token_uri: str = _TOKEN_URI,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_uri = token_uri
        self._app_folder_name = app_folder_name

    def connect(self) -> None:
        """No-op — no shared connection needed; each user gets their own client."""
        logger.info(
            "drive_service.ready",
            app_folder_name=self._app_folder_name,
        )

    def for_user(self, tokens: dict) -> UserDriveClient:
        """
        Build a ``UserDriveClient`` from the token dict stored in Redis.

        The token dict must contain at minimum ``token`` (access token) and
        ``refresh_token``.  The remaining fields (``token_uri``, ``client_id``,
        ``client_secret``, ``scopes``, ``expiry``) are merged with the
        service-level defaults so partial dicts still work.

        Args:
            tokens: Token dict previously returned by ``AuthService.exchange_code``
                    or ``AuthService.refresh_tokens`` and stored in Redis.

        Returns:
            ``UserDriveClient`` ready for Drive API calls.
        """
        import datetime

        expiry: Optional[datetime.datetime] = None
        raw_expiry = tokens.get("expiry")
        if raw_expiry:
            try:
                expiry = datetime.datetime.fromisoformat(raw_expiry)
            except (ValueError, TypeError):
                expiry = None

        credentials = Credentials(
            token=tokens.get("token"),
            refresh_token=tokens.get("refresh_token"),
            token_uri=tokens.get("token_uri", self._token_uri),
            client_id=tokens.get("client_id", self._client_id),
            client_secret=tokens.get("client_secret", self._client_secret),
            scopes=tokens.get("scopes", _DRIVE_SCOPES),
            expiry=expiry,
        )
        return UserDriveClient(
            credentials=credentials,
            app_folder_name=self._app_folder_name,
        )
