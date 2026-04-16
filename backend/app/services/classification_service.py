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
import math
import re
from typing import Any

import google.generativeai as genai

from backend.app.models.domain import ClassificationResult, DocType
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

# Number of leading characters sent for classification
_CLASSIFY_CHAR_LIMIT = 500

# Cosine similarity threshold for reusing an existing folder label
_LABEL_SIMILARITY_THRESHOLD = 0.85

_CLASSIFICATION_PROMPT = """Classify the following document excerpt.

Return ONLY valid JSON with no markdown, no explanation, no code fences:
{{"doc_type": "<one of: invoice, contract, letter, form, note, report, receipt, other>", "folder_label": "<2-4 word descriptive high-level category, e.g. 'Financial Invoice', 'Legal Contract', 'Machine Learning Research'>", "confidence": <float 0.0-1.0>}}

Document excerpt:
{text}"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class ClassificationService:
    """
    Single-purpose service for classifying a document's type via Gemini.

    Args:
        model:            Configured ``google.generativeai.GenerativeModel`` instance.
        embed_model_name: Gemini embedding model ID used for folder label normalization.
    """

    def __init__(self, model: Any, embed_model_name: str = "") -> None:
        self._model = model
        self._embed_model_name = embed_model_name

    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document from its text content.

        Only the first 500 characters are sent to minimise token usage.
        The response is parsed strictly — any deviation triggers a fallback
        to ``DocType.OTHER`` with ``confidence=0.0``.

        Args:
            text: Full or partial document text (may include [PAGE_BREAK] markers).

        Returns:
            ``ClassificationResult`` with ``doc_type``, ``folder_label``, and ``confidence``.
        """
        excerpt = text[:_CLASSIFY_CHAR_LIMIT].strip()
        if not excerpt:
            await logger.awarning("classification_service.empty_text")
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0, folder_label="Other Documents")

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
                folder_label=result.folder_label,
                confidence=result.confidence,
            )
            return result

        except Exception as exc:
            await logger.aerror(
                "classification_service.error",
                error=str(exc),
            )
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0, folder_label="Other Documents")

    async def normalize_label(
        self,
        new_label: str,
        existing_labels: list[str],
    ) -> str:
        """
        Normalize a new folder label against existing ones using embedding cosine similarity.

        If the most similar existing label exceeds ``_LABEL_SIMILARITY_THRESHOLD``,
        that existing label is returned so the document lands in the same folder.
        Otherwise ``new_label`` is returned as-is.

        Args:
            new_label:       Raw label returned by Gemini for the new document.
            existing_labels: All canonical labels already in use for this user.

        Returns:
            Canonical folder label string.
        """
        if not existing_labels or not self._embed_model_name:
            return new_label

        all_labels = [new_label] + existing_labels

        try:
            embeddings = await asyncio.to_thread(
                self._embed_labels, all_labels
            )
        except Exception as exc:
            await logger.awarning(
                "classification_service.normalize_label_embed_failed",
                error=str(exc),
            )
            return new_label

        new_vec = embeddings[0]
        best_sim = 0.0
        best_match = new_label

        for i, label in enumerate(existing_labels, start=1):
            sim = _cosine_similarity(new_vec, embeddings[i])
            if sim > best_sim:
                best_sim = sim
                best_match = label

        if best_sim >= _LABEL_SIMILARITY_THRESHOLD:
            await logger.ainfo(
                "classification_service.label_normalized",
                raw_label=new_label,
                canonical_label=best_match,
                similarity=round(best_sim, 3),
            )
            return best_match

        return new_label

    def _embed_labels(self, labels: list[str]) -> list[list[float]]:
        """Synchronous embedding call for a list of short label strings."""
        result = genai.embed_content(
            model=self._embed_model_name,
            content=labels,
            task_type="retrieval_document",
        )
        embedding = result["embedding"]
        # Normalise: single item → flat list; multiple → list of lists
        if labels and not isinstance(embedding[0], list):
            return [embedding]
        return embedding

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
            return ClassificationResult(doc_type=DocType.OTHER, confidence=0.0, folder_label="Other Documents")

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

        # folder_label: free-form descriptive name for the Drive folder
        folder_label = str(data.get("folder_label", "")).strip()
        if not folder_label:
            # Fallback: title-case the doc_type value
            folder_label = doc_type.value.replace("_", " ").title()

        # Sanitize: remove characters unsafe in Drive folder names
        folder_label = re.sub(r'[\\/:*?"<>|]', "", folder_label).strip()
        if not folder_label:
            folder_label = "Other Documents"

        return ClassificationResult(doc_type=doc_type, confidence=confidence, folder_label=folder_label)
