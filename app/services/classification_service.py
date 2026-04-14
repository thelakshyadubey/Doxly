"""
app/services/classification_service.py
=======================================
Document type classification via Gemini.

Sends the first 500 characters of the OCR text to the Gemini reasoning model
and expects a strict JSON response: ``{doc_type: string, confidence: float}``.

The classification result feeds into:
  - Session metadata (Redis)
  - Qdrant chunk payload filters
  - Neo4j Session node property
  - Drive folder hierarchy path
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import google.generativeai as genai

from app.models.domain import ClassificationResult, DocType
from app.utils.logger import get_logger
from app.utils.token_counter import truncate_to_tokens

logger = get_logger(__name__)

# Number of leading characters sent for classification
_CLASSIFY_CHAR_LIMIT = 500

_CLASSIFICATION_PROMPT = """Classify the following document excerpt as exactly one of these types:
[invoice, contract, letter, form, note, report, receipt, other]

Return ONLY valid JSON with no markdown, no explanation, no code fences:
{{"doc_type": "<type>", "confidence": <float 0.0-1.0>}}

Document excerpt:
{text}"""


class ClassificationService:
    """
    Single-purpose service for classifying a document's type via Gemini.

    Args:
        model: Configured ``google.generativeai.GenerativeModel`` instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document from its text content.

        Only the first 500 characters are sent to minimise token usage.
        The response is parsed strictly — any deviation triggers a fallback
        to ``DocType.OTHER`` with ``confidence=0.0``.

        Args:
            text: Full or partial document text (may include [PAGE_BREAK] markers).

        Returns:
            ``ClassificationResult`` with ``doc_type`` and ``confidence``.
        """
        excerpt = text[:_CLASSIFY_CHAR_LIMIT].strip()
        if not excerpt:
            await logger.awarning("classification_service.empty_text")
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0)

        prompt = _CLASSIFICATION_PROMPT.format(text=excerpt)

        try:
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
            )
            raw = response.text.strip()
            result = self._parse_response(raw)
            await logger.ainfo(
                "classification_service.classified",
                doc_type=result.doc_type,
                confidence=result.confidence,
            )
            return result

        except Exception as exc:
            await logger.aerror(
                "classification_service.error",
                error=str(exc),
            )
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> ClassificationResult:
        """
        Parse and validate the raw Gemini JSON response.

        Strips any accidental markdown fences before parsing.

        Args:
            raw: Raw string from Gemini.

        Returns:
            Validated ``ClassificationResult``.
        """
        # Strip markdown code fences if Gemini wraps the JSON anyway
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "classification_service.json_parse_error",
                raw=raw[:200],
                error=str(exc),
            )
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0)

        raw_type = str(data.get("doc_type", "other")).lower().strip()
        try:
            doc_type = DocType(raw_type)
        except ValueError:
            logger.warning(
                "classification_service.unknown_doc_type",
                raw_type=raw_type,
            )
            doc_type = DocType.OTHER

        raw_confidence = data.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        return ClassificationResult(doc_type=doc_type, confidence=confidence)
