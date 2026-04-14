"""
app/utils/id_generator.py
=========================
Deterministic UUID helpers.

All IDs in this system are derived deterministically from their semantic
components using UUID v5 (SHA-1 namespace hashing).  This guarantees that
re-ingesting the same document produces the same IDs, making upserts
idempotent across Qdrant and Neo4j.
"""

import hashlib
import math
import uuid

# Private namespace UUID used as the root for all domain IDs.
# Changing this invalidates all previously generated IDs.
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL


def _make_uuid5(*parts: str) -> str:
    """
    Derive a deterministic UUID v5 from one or more string parts.

    Parts are joined with ``|`` before hashing so that
    ``make_session_id("a", "bc")`` ≠ ``make_session_id("ab", "c")``.

    Args:
        *parts: Ordered string components that uniquely identify the object.

    Returns:
        Lowercase hyphenated UUID string.
    """
    combined = "|".join(parts)
    return str(uuid.uuid5(_NAMESPACE, combined))


def make_session_id(user_id: str, timestamp_bucket: int) -> str:
    """
    Generate the session ID for a given user and time bucket.

    Args:
        user_id:          Opaque user identifier.
        timestamp_bucket: ``floor(unix_timestamp / SESSION_THRESHOLD_SECONDS)``.

    Returns:
        Deterministic UUID string.
    """
    return _make_uuid5("session", user_id, str(timestamp_bucket))


def make_chunk_id(session_id: str, page_num: int, chunk_index: int) -> str:
    """
    Generate the chunk ID for a specific chunk within a page.

    Args:
        session_id:  Session UUID string.
        page_num:    1-based page number within the session document.
        chunk_index: 0-based index of the chunk within the page.

    Returns:
        Deterministic UUID string.
    """
    return _make_uuid5("chunk", session_id, str(page_num), str(chunk_index))


def make_document_id(session_id: str) -> str:
    """
    Generate the top-level document ID for a session.

    Args:
        session_id: Session UUID string.

    Returns:
        Deterministic UUID string.
    """
    return _make_uuid5("document", session_id)


def make_entity_id(canonical_name: str, entity_type: str) -> str:
    """
    Generate a stable entity ID from its canonical name and type.

    Args:
        canonical_name: Normalised entity name (e.g. "Acme Corp").
        entity_type:    Entity category string (e.g. "ORG").

    Returns:
        Deterministic UUID string.
    """
    return _make_uuid5("entity", canonical_name.lower().strip(), entity_type.lower().strip())


def compute_timestamp_bucket(unix_timestamp: float, threshold_seconds: int) -> int:
    """
    Compute the session time bucket for a given POSIX timestamp.

    All uploads whose timestamps fall in the same bucket share a session.

    Args:
        unix_timestamp:    POSIX timestamp (seconds since epoch).
        threshold_seconds: Session window duration from settings.

    Returns:
        Integer bucket index (``floor(unix_timestamp / threshold_seconds)``).
    """
    return math.floor(unix_timestamp / threshold_seconds)
