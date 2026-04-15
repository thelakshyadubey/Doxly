"""
app/utils/token_counter.py
==========================
Thin wrapper around tiktoken for accurate token counting.

Uses the ``cl100k_base`` encoding (same as GPT-4 / text-embedding-ada-002),
which is a reasonable proxy for Gemini token counts and matches the spec.
The encoder is lazily loaded and cached as a module-level singleton.
"""

from functools import lru_cache
from typing import Sequence

import tiktoken


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """
    Return the cached tiktoken encoder.

    ``cl100k_base`` is loaded once per process.  Call
    ``_get_encoder.cache_clear()`` in tests if needed.
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in *text* using cl100k_base encoding.

    Args:
        text: Input string to tokenise.

    Returns:
        Integer token count.
    """
    return len(_get_encoder().encode(text))


def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split *text* into overlapping token-bounded chunks.

    The function tokenises the full text once, then slides a window of
    ``chunk_size`` tokens with ``overlap`` tokens of carry-over, decoding
    each window back into a string.

    Args:
        text:       Input string to chunk.
        chunk_size: Maximum tokens per chunk (exclusive upper bound).
        overlap:    Number of tokens shared between consecutive chunks.

    Returns:
        Ordered list of decoded chunk strings.

    Raises:
        ValueError: If ``overlap`` >= ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    enc = _get_encoder()
    token_ids: list[int] = enc.encode(text)

    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_tokens = token_ids[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end == len(token_ids):
            break
        start += step

    return chunks


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate *text* so it fits within *max_tokens* tokens.

    Args:
        text:       Input string.
        max_tokens: Maximum token count for the returned string.

    Returns:
        The original string if it already fits, otherwise a decoded
        truncation of the first ``max_tokens`` token ids.
    """
    enc = _get_encoder()
    token_ids = enc.encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return enc.decode(token_ids[:max_tokens])
