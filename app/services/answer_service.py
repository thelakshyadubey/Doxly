"""
app/services/answer_service.py
================================
Gemini QnA with both streaming (SSE) and non-streaming modes.

Context assembly format per chunk:
    [SOURCE: <drive_path> | TYPE: <doc_type> | PAGE: <page_num>]
    <chunk_text>
    ---

Streaming:
    Yields SSE-formatted strings: ``data: {token}\\n\\n``
    Final event: ``data: [DONE]\\n\\n``

Non-streaming:
    Returns ``AnswerResult`` with answer text, citations, and heuristic confidence.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, AsyncIterator

from app.models.domain import AnswerResult, Citation, RankedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise document analyst. "
    "Answer only using information from the provided document context. "
    "If the answer is not present in the context, explicitly state that. "
    "Always cite the exact source path and page number for every claim."
)

_SENTINEL = object()  # signals end of stream from producer thread


class AnswerService:
    """
    Generates answers from retrieved context using Gemini.

    Args:
        model: Configured ``google.generativeai.GenerativeModel`` instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    # ── Non-streaming answer ──────────────────────────────────────────────────

    async def answer(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> AnswerResult:
        """
        Generate a complete non-streaming answer from retrieved chunks.

        Args:
            query:         The user's question.
            ranked_chunks: Ordered list from ``RetrievalService.retrieve``.

        Returns:
            ``AnswerResult`` with answer, citations, and heuristic confidence.
        """
        if not ranked_chunks:
            return AnswerResult(
                answer="No relevant documents were found for your query.",
                citations=[],
                confidence=0.0,
            )

        context_block = self._assemble_context(ranked_chunks)
        user_prompt = f"{context_block}\n\nQuestion: {query}"

        try:
            response = await asyncio.to_thread(
                self._model.generate_content,
                [
                    {"role": "user", "parts": [_SYSTEM_PROMPT + "\n\n" + user_prompt]},
                ],
            )
            answer_text = response.text.strip()
        except Exception as exc:
            await logger.aerror("answer_service.generation_failed", error=str(exc))
            raise

        citations = self._build_citations(ranked_chunks)
        confidence = self._heuristic_confidence(ranked_chunks)

        await logger.ainfo(
            "answer_service.answered",
            answer_chars=len(answer_text),
            citations=len(citations),
            confidence=round(confidence, 3),
        )

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            confidence=confidence,
        )

    # ── Streaming answer ──────────────────────────────────────────────────────

    async def answer_stream(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> AsyncIterator[str]:
        """
        Stream the Gemini response as Server-Sent Events.

        Yields SSE strings of the form ``data: {token}\\n\\n``.
        The final event is ``data: [DONE]\\n\\n``.

        The Gemini streaming iterator is synchronous, so it is consumed in a
        dedicated background thread that pushes tokens into a ``queue.Queue``.
        The async generator reads from the queue without blocking the event loop.

        Args:
            query:         The user's question.
            ranked_chunks: Ordered list from ``RetrievalService.retrieve``.

        Yields:
            SSE-formatted strings for use with FastAPI ``StreamingResponse``.
        """
        if not ranked_chunks:
            yield "data: No relevant documents were found for your query.\n\n"
            yield "data: [DONE]\n\n"
            return

        context_block = self._assemble_context(ranked_chunks)
        user_prompt = f"{context_block}\n\nQuestion: {query}"
        full_prompt = _SYSTEM_PROMPT + "\n\n" + user_prompt

        token_queue: queue.Queue[object] = queue.Queue()
        model = self._model

        def _produce() -> None:
            """Run Gemini streaming in a thread, push tokens to the queue."""
            try:
                stream = model.generate_content(full_prompt, stream=True)
                for chunk in stream:
                    token = getattr(chunk, "text", None)
                    if token:
                        token_queue.put(token)
            except Exception as exc:  # noqa: BLE001
                token_queue.put(exc)
            finally:
                token_queue.put(_SENTINEL)

        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        try:
            while True:
                # Poll the queue asynchronously so we don't block the event loop.
                item = await loop.run_in_executor(None, token_queue.get)
                if item is _SENTINEL:
                    break
                if isinstance(item, Exception):
                    await logger.aerror("answer_service.stream_failed", error=str(item))
                    yield f"data: [ERROR] {item}\n\n"
                    break
                yield f"data: {item}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    # ── Context assembly ──────────────────────────────────────────────────────

    def _assemble_context(self, ranked_chunks: list[RankedChunk]) -> str:
        """
        Build the formatted context block passed to Gemini.

        Args:
            ranked_chunks: Ordered list of retrieved chunks.

        Returns:
            Multi-chunk context string.
        """
        parts: list[str] = []
        for rc in ranked_chunks:
            header = (
                f"[SOURCE: {rc.source_drive_path} | "
                f"TYPE: {rc.doc_type} | "
                f"PAGE: {rc.page_num}]"
            )
            parts.append(f"{header}\n{rc.chunk_text}\n---")
        return "\n\n".join(parts)

    def _build_citations(self, ranked_chunks: list[RankedChunk]) -> list[Citation]:
        """
        Build citation objects from ranked chunks.

        Args:
            ranked_chunks: Top-k chunks after RRF.

        Returns:
            List of ``Citation`` domain objects.
        """
        return [
            Citation(
                source=rc.source_drive_path,
                page=rc.page_num,
                chunk_id=rc.chunk_id,
            )
            for rc in ranked_chunks
        ]

    def _heuristic_confidence(self, ranked_chunks: list[RankedChunk]) -> float:
        """
        Estimate answer confidence from the top RRF score.

        Maps the top RRF score to [0.0, 1.0] using a simple normalisation.
        The absolute range of RRF scores depends on RETRIEVAL_TOP_K and
        FINAL_TOP_K, so this is an approximation only.

        Args:
            ranked_chunks: Ordered list (highest RRF first).

        Returns:
            Float in [0.0, 1.0].
        """
        if not ranked_chunks:
            return 0.0
        top_score = ranked_chunks[0].rrf_score
        # Normalise: a perfect single-source rank-1 hit scores 1/(60+1) ≈ 0.016
        # Two sources at rank 1 → ≈ 0.033.  Cap at 0.1 as "max plausible".
        return min(top_score / 0.1, 1.0)
