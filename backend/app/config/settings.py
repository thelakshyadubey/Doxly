"""
app/config/settings.py
======================
Single source of truth for all runtime configuration.
Reads from environment variables / .env file via Pydantic BaseSettings.
No value is hardcoded here — every field has a corresponding .env key.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolves to backend/.env regardless of where uvicorn is launched from
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """
    Aggregates all environment-driven configuration for the application.

    Field names match .env keys exactly (case-insensitive by Pydantic convention).
    """

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    gemini_reasoning_model: str = Field(
        default="gemini-1.5-pro",
        description="Gemini model ID used for classification, coref, and QnA",
    )
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini model ID used for vector embeddings",
    )

    # ── Google OAuth 2.0 (per-user Drive access) ──────────────────────────────
    google_oauth_client_id: str = Field(
        ..., description="OAuth 2.0 client ID from Google Cloud Console"
    )
    google_oauth_client_secret: str = Field(
        ..., description="OAuth 2.0 client secret from Google Cloud Console"
    )
    google_oauth_redirect_uri: str = Field(
        default="http://localhost:8000/auth/callback",
        description="Callback URL registered in Google Cloud Console",
    )
    google_drive_app_folder_name: str = Field(
        default="DocumentIntelligence",
        description=(
            "Top-level folder created in each user's own Google Drive. "
            "Structure: My Drive / {app_folder_name} / {session_id} / {doc_type} /"
        ),
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(
        default="",
        description="Qdrant cloud URL. Leave empty to use local persistent mode.",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant cloud API key. Only required when qdrant_url is set.",
    )
    qdrant_collection_name: str = Field(default="documents")
    qdrant_local_path: str = Field(
        default="./qdrant_data",
        description="Filesystem path for local Qdrant storage when qdrant_url is empty.",
    )
    qdrant_vector_size: int = Field(
        default=3072,
        description="Embedding dimension — must match the configured embedding model output.",
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(..., description="Neo4j Bolt/AuraDB connection URI")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(..., description="Neo4j authentication password")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(..., description="Redis connection URL (Upstash or local)")

    # ── Session windowing ─────────────────────────────────────────────────────
    session_threshold_seconds: int = Field(
        default=600,
        description="Duration (seconds) of a session upload window.",
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    embedding_batch_size: int = Field(default=32, gt=0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=8)
    final_top_k: int = Field(default=6)

    # ── App ───────────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    frontend_url: str = Field(
        default="http://localhost:5173",
        description="Frontend origin — OAuth callback redirects here after authorization.",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a recognised Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        """Ensure overlap does not exceed chunk size."""
        chunk_size = info.data.get("chunk_size_tokens")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be less than chunk_size_tokens ({chunk_size})"
            )
        return v

    @property
    def qdrant_is_local(self) -> bool:
        """True when running Qdrant in local persistent mode."""
        return not self.qdrant_url.strip()

    @property
    def oauth_client_config(self) -> dict:
        """
        Return the OAuth 2.0 client config dict expected by google-auth-oauthlib Flow.

        This format matches the JSON structure of a downloaded OAuth client secret file.
        """
        return {
            "web": {
                "client_id": self.google_oauth_client_id,
                "client_secret": self.google_oauth_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.google_oauth_redirect_uri],
            }
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached singleton Settings instance.

    Using lru_cache ensures .env is parsed exactly once per process.
    Call ``get_settings.cache_clear()`` in tests to force re-parse.
    """
    return Settings()
