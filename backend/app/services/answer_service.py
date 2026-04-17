"""
app/services/answer_service.py
================================
Gemini QnA — non-streaming only.

Context assembly format per chunk:
    [SOURCE: <drive_path> | PAGE: <page_num>]
    <chunk_text>
    ---
"""

from __future__ import annotations

import asyncio
from typing import Any

from backend.app.models.domain import AnswerResult, Citation, RankedChunk
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a precise document analyst. "
    "Answer only using information from the provided document context. "
    "If the answer is not present in the context, explicitly state that. "
    "Always cite the exact source path and page number for every claim."
)

_NO_CONTEXT_MSG = (
    "No relevant documents were found for your query. "
    "Please upload documents first, then ask questions about them."
)


class AnswerService:
    """
    Generates answers from retrieved context using Gemini.

    Args:
        model: Configured ``google.generativeai.GenerativeModel`` instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    async def answer(
        self,
        query: str,
        ranked_chunks: list[RankedChunk],
    ) -> AnswerResult:
        """
        Generate a complete answer from retrieved chunks.

        Args:
            query:         The user's question.
            ranked_chunks: Ordered list from ``RetrievalService.retrieve``.

        Returns:
            ``AnswerResult`` with answer, citations, and heuristic confidence.
        """
        if not ranked_chunks:
            return AnswerResult(answer=_NO_CONTEXT_MSG, citations=[], confidence=0.0)

        context_block = self._assemble_context(ranked_chunks)
        full_prompt = _SYSTEM_PROMPT + "\n\n" + context_block + "\n\nQuestion: " + query

        try:
            response = await asyncio.to_thread(
                self._model.generate_content, full_prompt
            )
            answer_text = _safe_text(response)
        except Exception as exc:
            await logger.aerror("answer_service.generation_failed", error=str(exc))
            raise

        if not answer_text:
            answer_text = "Gemini returned an empty or blocked response. Please try rephrasing your question."

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

    def _assemble_context(self, ranked_chunks: list[RankedChunk]) -> str:
        parts: list[str] = []
        for rc in ranked_chunks:
            header = f"[SOURCE: {rc.source_drive_path} | PAGE: {rc.page_num}]"
            parts.append(f"{header}\n{rc.chunk_text}\n---")
        return "\n\n".join(parts)

    def _build_citations(self, ranked_chunks: list[RankedChunk]) -> list[Citation]:
        return [
            Citation(
                source=rc.source_drive_path,
                page=rc.page_num,
                chunk_id=rc.chunk_id,
            )
            for rc in ranked_chunks
        ]

    def _heuristic_confidence(self, ranked_chunks: list[RankedChunk]) -> float:
        if not ranked_chunks:
            return 0.0
        top_score = ranked_chunks[0].rrf_score
        return min(top_score / 0.1, 1.0)


def _safe_text(response: Any) -> str:
    """Safely extract text from a Gemini response, falling back to walking parts."""
    try:
        text = response.text
        if text:
            return text.strip()
    except Exception:
        pass

    try:
        parts: list[str] = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
        text = "".join(parts).strip()
        if text:
            return text
    except Exception:
        pass

    return ""
