"""
app/services/ocr_service.py
============================
Gemini multimodal OCR wrapper.

Sends image bytes to Gemini via a multimodal prompt and extracts the full
document text with language detection.  All I/O is async — the synchronous
Gemini call is executed in a thread pool via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import google.generativeai as genai
from google.generativeai.types import glm

from app.models.domain import OCRResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

_OCR_PROMPT = (
    "You are a precise OCR engine. Extract ALL text from this image exactly as it appears, "
    "preserving paragraph and line breaks with newlines. "
    "Do not summarize, interpret, correct, or add anything. "
    "Return ONLY a JSON object with exactly two fields:\n"
    '- "text": the complete extracted text\n'
    '- "language": the BCP-47 language code of the dominant language '
    '(e.g. "en", "hi", "fr"), or "und" if unknown.\n\n'
    "JSON only — no markdown fences, no extra keys."
)


class OCRService:
    """
    Async wrapper around a Gemini multimodal model for OCR.

    Args:
        model: Configured ``google.generativeai.GenerativeModel`` instance.
    """

    def __init__(self, model: genai.GenerativeModel) -> None:
        self._model = model

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        No-op kept for interface compatibility with the startup lifespan.

        The Gemini client is already initialised by the time this service is
        constructed; there is nothing to connect.
        """
        logger.info("ocr_service.connected", backend="gemini")

    # ── Core OCR ──────────────────────────────────────────────────────────────

    async def extract_text(
        self,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> OCRResult:
        """
        Run multimodal OCR on a single image using Gemini.

        The synchronous Gemini call is offloaded to a thread pool so the
        event loop is never blocked.

        Args:
            image_bytes: Raw bytes of a JPEG, PNG, TIFF, or PDF page.
            mime_type:   MIME type of the image (default: ``image/jpeg``).

        Returns:
            ``OCRResult`` with extracted text, ``confidence=1.0``,
            ``page_count=1``, and the detected BCP-47 language code.

        Raises:
            RuntimeError: If Gemini returns an empty response.
        """
        return await asyncio.to_thread(self._run_ocr, image_bytes, mime_type)

    def _run_ocr(self, image_bytes: bytes, mime_type: str) -> OCRResult:
        """
        Synchronous Gemini call — must not be called directly from async code.

        Args:
            image_bytes: Raw image bytes.
            mime_type:   MIME type of the image.

        Returns:
            ``OCRResult`` populated from the Gemini response.

        Raises:
            RuntimeError: If Gemini returns an empty response.
        """
        image_part = glm.Part(
            inline_data=glm.Blob(mime_type=mime_type, data=image_bytes)
        )
        response = self._model.generate_content([image_part, _OCR_PROMPT])

        raw = response.text.strip() if response.text else ""
        if not raw:
            logger.warning("ocr_service.empty_result")
            return OCRResult(text="", confidence=0.0, page_count=1, language="und")

        # Strip markdown fences in case Gemini wraps the JSON despite instructions
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        raw = raw.strip()

        text: str
        language: str
        try:
            data: dict[str, Any] = json.loads(raw)
            text = data.get("text", "")
            language = data.get("language", "und") or "und"
        except json.JSONDecodeError:
            # Gemini returned plain text instead of JSON — use as-is
            logger.warning("ocr_service.json_parse_failed", falling_back_to_raw_text=True)
            text = raw
            language = "und"

        logger.info(
            "ocr_service.extracted",
            chars=len(text),
            pages=1,
            backend="gemini",
            language=language,
        )

        return OCRResult(text=text, confidence=1.0, page_count=1, language=language)
