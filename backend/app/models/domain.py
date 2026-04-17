"""
app/models/domain.py
====================
Internal domain models shared across services and stores.

All models use Pydantic v2.  None of these are exposed directly as API
request/response schemas — see ``app/models/api.py`` for that layer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────


class ChunkRole(str, Enum):
    """Distinguishes parent page chunks from child sliding-window chunks."""

    PARENT = "parent"
    CHILD = "child"


class SessionStatus(str, Enum):
    """Lifecycle states of an upload session."""

    OPEN = "open"          # accepting page uploads
    QUEUED = "queued"      # window closed, pipeline not yet triggered
    PROCESSING = "processing"
    INDEXED = "indexed"    # fully ingested into Qdrant + Neo4j
    FAILED = "failed"


# ──────────────────────────────────────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────────────────────────────────────


class OCRResult(BaseModel):
    """Output produced by the OCR service for a single image page."""

    text: str = Field(..., description="Full extracted text from the image")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate OCR confidence score [0, 1]"
    )
    page_count: int = Field(default=1, ge=1)
    language: str = Field(default="en", description="BCP-47 language code detected by OCR")


# ──────────────────────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────────────────────


class SessionMetadata(BaseModel):
    """
    Redis-persisted metadata for an active upload session.

    Key: ``session:{user_id}:{session_id}``
    TTL: ``SESSION_THRESHOLD_SECONDS``
    """

    user_id: str
    session_id: str
    drive_folder_id: str = Field(default="", description="Drive folder created for this session")
    uploaded_file_ids: list[str] = Field(
        default_factory=list,
        description="Drive file IDs of images uploaded during the upload phase",
    )
    page_count: int = Field(default=0, ge=0)
    status: SessionStatus = SessionStatus.OPEN
    accumulated_text: str = Field(
        default="",
        description="Running concatenation of OCR output, pages separated by [PAGE_BREAK]",
    )
    timestamp_bucket: int = Field(
        ..., description="floor(upload_time / SESSION_THRESHOLD_SECONDS)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entities
# ──────────────────────────────────────────────────────────────────────────────


class EntityMention(BaseModel):
    """A single named entity extracted during the first coref-resolution pass."""

    canonical: str = Field(..., description="Normalised canonical name for the entity")
    aliases: list[str] = Field(default_factory=list, description="All surface forms found in text")
    type: str = Field(..., description="Entity type label, e.g. ORG, PERSON, DATE, MONEY")
    first_mention_offset: int = Field(
        default=0, ge=0, description="Character offset of the first occurrence in the source text"
    )


class EntityMap(BaseModel):
    """Collection of all entities extracted from a document."""

    entities: list[EntityMention] = Field(default_factory=list)

    @property
    def canonical_names(self) -> list[str]:
        """Return deduplicated list of canonical entity names."""
        return list({e.canonical for e in self.entities})


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────


class ClassificationResult(BaseModel):
    """Output from the classification service."""

    confidence: float = Field(..., ge=0.0, le=1.0)
    folder_label: str = Field(
        default="",
        description="Free-form 2-4 word descriptive name used as the Drive folder label",
    )
    topic_summary: str = Field(
        default="",
        description=(
            "1-2 sentence description of what the document is fundamentally about. "
            "Used by the routing step to match against existing folder topics."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Chunks
# ──────────────────────────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """
    A single text chunk ready for embedding and storage.

    Parent chunks represent whole pages; child chunks are sliding-window
    sub-divisions of a parent with an entity-context header prepended.
    """

    chunk_id: str = Field(..., description="Deterministic UUID from id_generator")
    session_id: str
    user_id: str
    page_num: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    role: ChunkRole
    text: str = Field(..., description="Raw chunk text (no entity header)")
    enriched_text: str = Field(
        ..., description="Text with entity header prepended, used for embedding"
    )
    parent_chunk_id: Optional[str] = Field(
        default=None, description="chunk_id of the parent page chunk (None for parents)"
    )
    entity_map: EntityMap = Field(default_factory=EntityMap)
    source_drive_path: str = Field(default="")


class EmbeddedChunk(BaseModel):
    """A Chunk paired with its embedding vector, ready for Qdrant upsert."""

    chunk: Chunk
    vector: list[float] = Field(..., description="Embedding of chunk.enriched_text")


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────


class ScoredChunk(BaseModel):
    """A chunk returned from Qdrant search, augmented with its relevance score."""

    chunk_id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class RankedChunk(BaseModel):
    """A chunk after Reciprocal Rank Fusion re-ranking, ready for context assembly."""

    chunk_id: str
    rrf_score: float
    chunk_text: str
    source_drive_path: str
    page_num: int


# ──────────────────────────────────────────────────────────────────────────────
# Answer
# ──────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """Source attribution for a claim in the generated answer."""

    source: str = Field(..., description="Drive path of the source document")
    page: int
    chunk_id: str


class AnswerResult(BaseModel):
    """Non-streaming answer returned by the answer service."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heuristic confidence derived from top RRF score",
    )
