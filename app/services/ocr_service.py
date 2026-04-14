"""
app/services/ocr_service.py
============================
Google Cloud Vision API wrapper for layout-aware OCR.

Sends image bytes via ``DOCUMENT_TEXT_DETECTION`` and extracts the
full_text_annotation, which preserves paragraph and page structure.
All I/O is async — the synchronous Vision client is executed in a
thread pool via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from google.cloud import vision

from app.models.domain import OCRResult
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OCRService:
    """
    Thin async wrapper around the Google Cloud Vision ``ImageAnnotatorClient``.

    The Vision client is instantiated once and reused.  Authentication is
    handled automatically by the Google auth library using the credentials
    pointed to by ``GOOGLE_APPLICATION_CREDENTIALS``.

    Args:
        project_id: GCP project ID (used for request attribution / billing).
    """

    def __init__(self, project_id: str) -> None:
        self._project_id = project_id
        self._client: Optional[vision.ImageAnnotatorClient] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Instantiate the Vision API client.

        Must be called once during app lifespan startup.
        """
        self._client = vision.ImageAnnotatorClient()
        logger.info("ocr_service.connected", project=self._project_id)

    # ── Core OCR ──────────────────────────────────────────────────────────────

    async def extract_text(self, image_bytes: bytes) -> OCRResult:
        """
        Run ``DOCUMENT_TEXT_DETECTION`` on a single image and return structured output.

        The synchronous Vision API call is offloaded to a thread pool so the
        event loop is never blocked.

        Args:
            image_bytes: Raw bytes of a JPEG, PNG, TIFF, or PDF page.

        Returns:
            ``OCRResult`` with extracted text, aggregate confidence, page count,
            and detected language.

        Raises:
            RuntimeError: If the Vision API returns an error response.
        """
        result = await asyncio.to_thread(self._run_ocr, image_bytes)
        return result

    def _run_ocr(self, image_bytes: bytes) -> OCRResult:
        """
        Synchronous Vision API call — must not be called directly from async code.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            ``OCRResult`` populated from the Vision API response.

        Raises:
            RuntimeError: On Vision API error.
        """
        image = vision.Image(content=image_bytes)
        response = self._client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(
                f"Vision API error: {response.error.message} "
                f"(code {response.error.code})"
            )

        full_annotation = response.full_text_annotation

        if not full_annotation or not full_annotation.text:
            logger.warning("ocr_service.empty_result")
            return OCRResult(text="", confidence=0.0, page_count=1, language="und")

        # Aggregate confidence from all blocks across all pages
        confidences: list[float] = []
        page_count = len(full_annotation.pages)
        page_texts: list[str] = []

        for page in full_annotation.pages:
            for block in page.blocks:
                if block.confidence > 0:
                    confidences.append(block.confidence)
            # Reconstruct per-page text from paragraph text segments
            page_text_parts: list[str] = []
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    para_text = "".join(
                        "".join(symbol.text for symbol in word.symbols)
                        for word in paragraph.words
                    )
                    if para_text:
                        page_text_parts.append(para_text)
            page_texts.append(" ".join(page_text_parts))

        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # For multi-page documents insert [PAGE_BREAK] between pages;
        # for single-page use full_text_annotation.text directly (preserves layout).
        if page_count > 1:
            full_text = "\n[PAGE_BREAK]\n".join(page_texts)
        else:
            full_text = full_annotation.text

        # Detect dominant language from the first page's detected languages
        language = "und"
        if full_annotation.pages:
            first_page = full_annotation.pages[0]
            if first_page.property and first_page.property.detected_languages:
                language = first_page.property.detected_languages[0].language_code

        logger.info(
            "ocr_service.extracted",
            chars=len(full_text),
            pages=page_count,
            confidence=round(avg_confidence, 3),
            language=language,
        )

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            page_count=max(page_count, 1),
            language=language or "und",
        )
