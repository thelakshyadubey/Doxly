"""
app/services/classification_service.py
=======================================
Document type classification and intelligent folder routing via Gemini.

Two-step process
----------------
Step 1 — classify(text):
    Sends the first 1 500 characters to Gemini.
    Returns doc_type, a short folder_label, and a topic_summary (1-2 sentences
    describing the document's core subject matter).

Step 2 — route_to_folder(topic_summary, raw_label, folder_registry):
    Given the existing folder registry {name: topic_summary} for this user,
    asks Gemini to pick the best matching folder or declare NEW.

    For large registries (> _MAX_DIRECT_ROUTE_FOLDERS) an embedding pre-filter
    narrows to the top-5 candidates before the Gemini call, keeping the prompt
    small.

Why Gemini routing instead of cosine similarity on label strings?
    Short label strings like "AI Research Summary" and "Technical Study Notes"
    have low cosine similarity even when the underlying documents are about the
    same topic (e.g. RAG).  Gemini understands that "Retrieval Augmented
    Generation" and "RAG" are the same thing; cosine similarity of two 3-word
    labels does not.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

import google.generativeai as genai

from backend.app.models.domain import ClassificationResult
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

# Leading characters sent to Gemini for classification.
# 1 500 chars captures more document substance than the old 500-char limit,
# which was too narrow for documents whose topic only becomes clear after the
# opening paragraph.  classify() already runs in parallel with coref (which
# is far slower), so the extra tokens don't affect wall-clock time.
_CLASSIFY_CHAR_LIMIT = 1500

# When the registry has more folders than this, run an embedding pre-filter
# first (pick top-5 by cosine similarity) to keep the Gemini routing prompt
# small regardless of how many folders the user has accumulated.
_MAX_DIRECT_ROUTE_FOLDERS = 10

_CLASSIFICATION_PROMPT = """\
Describe the core topic of the following document excerpt.

Return ONLY valid JSON — no markdown, no explanation, no code fences:
{{
  "topic_summary": "<1-2 sentences describing what this document is fundamentally about — \
the core subject matter, concepts, and domain. Be specific enough that a filing assistant \
could recognise another document on the same topic even if it uses different terminology.>",
  "folder_label": "<The most specific name for this document's subject — use the exact \
technology name, acronym, company name, or precise subject that best identifies it. \
RULES: (1) If the document is about a named concept or technology, use that name directly \
— 'RAG' not 'AI Research', 'BERT' not 'NLP Paper', 'Transformers' not 'Deep Learning'. \
(2) If there is a company or entity involved, lead with it — 'Amazon Invoices' not \
'Financial Documents', 'HDFC Statements' not 'Bank Records', 'Google NDA' not \
'Legal Documents'. (3) If it is a standard document type with no specific entity, use \
the precise type — 'Tax Returns', 'Lab Reports', 'Prescriptions', 'Salary Slips'. \
(4) Never use vague category words like 'Documents', 'Records', 'Study', 'Research', \
'Notes' unless there is truly nothing more specific. Maximum 3 words.>",
  "confidence": <float 0.0-1.0>
}}

Document excerpt:
{text}"""

_ROUTING_PROMPT = """\
You are a document filing assistant. Decide which existing folder this new document \
belongs in, based on its topic.

New document topic:
{topic_summary}

Existing folders:
{folder_list}

Rules:
1. If this document is about the same subject as an existing folder — even if the \
terminology differs (e.g. "RAG" and "Retrieval Augmented Generation" are the same topic) \
— reply with EXACTLY that folder name, copied character-for-character from the list above.
2. If none of the existing folders are a good match, reply with exactly: NEW
3. Reply with ONLY the folder name or the word NEW. No explanation, no quotes, no \
punctuation around the name."""


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
    Classifies documents and routes them to the correct Drive folder.

    Args:
        model:            Configured ``google.generativeai.GenerativeModel`` instance.
        embed_model_name: Gemini embedding model ID used for pre-filtering large
                          registries.  May be empty — routing still works via
                          Gemini alone, just without the pre-filter step.
    """

    def __init__(self, model: Any, embed_model_name: str = "") -> None:
        self._model = model
        self._embed_model_name = embed_model_name

    # ── Step 1: classify ──────────────────────────────────────────────────────

    async def classify(self, text: str) -> ClassificationResult:
        """
        Classify a document from its text content.

        Only the first ``_CLASSIFY_CHAR_LIMIT`` characters are sent to reduce
        token usage.  The response is parsed strictly — any deviation triggers
        a fallback to ``DocType.OTHER`` with ``confidence=0.0``.

        Args:
            text: Full or partial document text (may include [PAGE_BREAK] markers).

        Returns:
            ``ClassificationResult`` with ``doc_type``, ``folder_label``,
            ``topic_summary``, and ``confidence``.
        """
        excerpt = text[:_CLASSIFY_CHAR_LIMIT].strip()
        if not excerpt:
            await logger.awarning("classification_service.empty_text")
            return ClassificationResult(
                confidence=0.0,
                folder_label="Other Documents",
                topic_summary="",
            )

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
                folder_label=result.folder_label,
                topic_summary=result.topic_summary[:120],
                confidence=result.confidence,
            )
            return result

        except Exception as exc:
            await logger.aerror(
                "classification_service.error",
                error=str(exc),
            )
            return ClassificationResult(
                confidence=0.0,
                folder_label="Other Documents",
                topic_summary="",
            )

    # ── Step 2: route to folder ───────────────────────────────────────────────

    async def route_to_folder(
        self,
        topic_summary: str,
        raw_label: str,
        folder_registry: dict[str, str],
    ) -> str:
        """
        Pick the best existing folder for this document, or return ``raw_label``
        to signal that a new folder should be created.

        Uses Gemini to understand semantic equivalence between topics (e.g.
        "RAG" == "Retrieval Augmented Generation").  For registries larger than
        ``_MAX_DIRECT_ROUTE_FOLDERS``, an embedding pre-filter narrows to the
        top-5 candidates first.

        Args:
            topic_summary:   1-2 sentence description from ``classify()``.
            raw_label:       Folder label also from ``classify()`` — used as the
                             new folder name if Gemini returns NEW.
            folder_registry: {folder_name: topic_summary} for all existing folders
                             belonging to this user.

        Returns:
            An existing folder name, or ``raw_label`` if no match was found.
        """
        if not folder_registry:
            return raw_label

        # For large registries, pre-filter to top-5 by embedding similarity.
        candidates = folder_registry
        if len(folder_registry) > _MAX_DIRECT_ROUTE_FOLDERS and self._embed_model_name:
            candidates = await self._prefilter_folders(topic_summary, folder_registry)

        if not candidates:
            return raw_label

        # Build the folder list shown to Gemini.
        folder_list = "\n".join(
            f'- "{name}": {summary}' for name, summary in candidates.items()
        )
        prompt = _ROUTING_PROMPT.format(
            topic_summary=topic_summary or raw_label,
            folder_list=folder_list,
        )

        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            answer = response.text.strip().strip('"').strip("'").strip()
        except Exception as exc:
            await logger.awarning(
                "classification_service.routing_gemini_failed",
                error=str(exc),
            )
            return raw_label

        if answer == "NEW":
            await logger.ainfo(
                "classification_service.new_folder",
                raw_label=raw_label,
                topic_summary=topic_summary[:120],
            )
            return raw_label

        # Case-insensitive match to absorb minor Gemini casing drift.
        lower_map = {k.lower(): k for k in folder_registry}
        matched = lower_map.get(answer.lower())

        if matched:
            await logger.ainfo(
                "classification_service.folder_routed",
                routed_to=matched,
                raw_label=raw_label,
                topic_summary=topic_summary[:120],
            )
            return matched

        # Gemini returned something that isn't a known folder name — treat as NEW.
        await logger.awarning(
            "classification_service.routing_unknown_answer",
            answer=answer,
            raw_label=raw_label,
        )
        return raw_label

    # ── Embedding pre-filter ──────────────────────────────────────────────────

    async def _prefilter_folders(
        self,
        topic_summary: str,
        folder_registry: dict[str, str],
        top_n: int = 5,
    ) -> dict[str, str]:
        """
        Narrow a large folder registry to the ``top_n`` most semantically similar
        folders using embedding cosine similarity.

        This keeps the Gemini routing prompt small regardless of how many folders
        the user has accumulated.

        Args:
            topic_summary:   Topic of the incoming document.
            folder_registry: Full {name: topic_summary} registry.
            top_n:           Maximum candidates to return.

        Returns:
            Subset of ``folder_registry`` with the best-matching folders.
            Falls back to the first ``top_n`` entries if embedding fails.
        """
        names = list(folder_registry.keys())
        folder_topics = [folder_registry[n] for n in names]
        all_texts = [topic_summary] + folder_topics

        try:
            embeddings = await asyncio.to_thread(self._embed_texts, all_texts)
        except Exception as exc:
            await logger.awarning(
                "classification_service.prefilter_embed_failed",
                error=str(exc),
            )
            return dict(list(folder_registry.items())[:top_n])

        query_vec = embeddings[0]
        scored = [
            (_cosine_similarity(query_vec, embeddings[i + 1]), name)
            for i, name in enumerate(names)
        ]
        scored.sort(reverse=True)
        top_names = [name for _, name in scored[:top_n]]
        return {name: folder_registry[name] for name in top_names}

    # ── Shared embedding helper ───────────────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings using the Gemini embedding model.

        Args:
            texts: Strings to embed.

        Returns:
            List of float vectors in the same order as *texts*.
        """
        result = genai.embed_content(
            model=self._embed_model_name,
            content=texts,
            task_type="retrieval_document",
        )
        embedding = result["embedding"]
        # Normalise: single item → flat list; multiple → list of lists
        if texts and not isinstance(embedding[0], list):
            return [embedding]
        return embedding

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> ClassificationResult:
        """
        Parse and validate the raw Gemini JSON response from ``classify()``.

        Args:
            raw: Raw string from Gemini.

        Returns:
            Validated ``ClassificationResult``.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "classification_service.json_parse_error",
                raw=raw[:200],
                error=str(exc),
            )
            return ClassificationResult(
                confidence=0.0,
                folder_label="Other Documents",
                topic_summary="",
            )

        raw_confidence = data.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        # topic_summary — the core semantic description used for routing
        topic_summary = str(data.get("topic_summary", "")).strip()

        # folder_label — short human-visible Drive folder name
        folder_label = str(data.get("folder_label", "")).strip()
        if not folder_label:
            folder_label = "General"

        # Sanitize: remove characters unsafe in Drive folder names
        folder_label = re.sub(r'[\\/:*?"<>|]', "", folder_label).strip()
        if not folder_label:
            folder_label = "Other Documents"

        return ClassificationResult(
            confidence=confidence,
            folder_label=folder_label,
            topic_summary=topic_summary,
        )
