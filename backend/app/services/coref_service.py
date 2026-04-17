"""
app/services/coref_service.py
==============================
Two-pass coreference resolution via Gemini.

Pass 1 — Entity extraction:
    Send full text → receive EntityMap (canonical names, aliases, types, offsets).

Pass 2 — Pronoun / reference replacement:
    Send full text + EntityMap → receive text with all pronouns replaced by
    their canonical entity names, preserving all original structure.

The resolved text and EntityMap are then forwarded to the chunking service.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from backend.app.models.domain import EntityMap, EntityMention
from backend.app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_ENTITY_EXTRACTION_PROMPT = """Extract all named entities and their aliases from the document below.
Return ONLY valid JSON with no markdown, no explanation, no code fences:
{{"entities": [
  {{
    "canonical": "<primary name>",
    "aliases": ["<alias1>", "<alias2>"],
    "type": "<ORG|PERSON|DATE|MONEY|PLACE|OTHER>",
    "first_mention_offset": <integer character offset>
  }}
]}}

Document:
{text}"""

_COREF_RESOLUTION_PROMPT = """Using the entity map below, replace every pronoun (it, they, he, she, \
this, that, these, those) and every ambiguous reference with its canonical entity name from the map.

CRITICAL RULES — violating these makes the output unusable:
1. Do NOT alter any parenthetical that immediately follows an acronym or proper noun and contains its \
definition or expansion. For example: "OASIS (open agent social interaction simulation)" must remain \
exactly as written — never rewrite it as "OASIS (OASIS)".
2. Only resolve genuine pronouns and ambiguous references (words like "it", "they", "this system", etc.) \
that refer back to an entity mentioned earlier.
3. Preserve all original punctuation, line breaks, spacing, and [PAGE_BREAK] markers exactly.
Return ONLY the resolved text — no explanation, no JSON, no code fences.

Entity map:
{entity_map_json}

Original text:
{text}"""


class CorefService:
    """
    Performs two-pass coreference resolution using the Gemini reasoning model.

    Args:
        model: Configured ``google.generativeai.GenerativeModel`` instance.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    # ── Public API ────────────────────────────────────────────────────────────

    async def resolve(self, text: str) -> tuple[str, EntityMap]:
        """
        Run both passes and return the resolved text with its entity map.

        Args:
            text: Full document text, pages separated by ``[PAGE_BREAK]``.

        Returns:
            A tuple of ``(resolved_text, entity_map)``.
            Falls back to ``(original_text, empty_entity_map)`` on any error.
        """
        entity_map = await self._extract_entities(text)
        resolved_text = await self._resolve_references(text, entity_map)
        return resolved_text, entity_map

    # ── Pass 1: entity extraction ─────────────────────────────────────────────

    async def _extract_entities(self, text: str) -> EntityMap:
        """
        Pass 1 — extract all named entities from the document.

        Args:
            text: Full document text.

        Returns:
            ``EntityMap`` populated with extracted entities.
            Returns an empty ``EntityMap`` on failure.
        """
        prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text)
        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            raw = response.text.strip()
            entity_map = self._parse_entity_map(raw)
            await logger.ainfo(
                "coref_service.entities_extracted",
                count=len(entity_map.entities),
            )
            return entity_map
        except Exception as exc:
            await logger.aerror("coref_service.entity_extraction_failed", error=str(exc))
            return EntityMap()

    def _parse_entity_map(self, raw: str) -> EntityMap:
        """
        Parse the Gemini JSON response for Pass 1.

        Args:
            raw: Raw string from Gemini.

        Returns:
            Validated ``EntityMap``.
        """
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "coref_service.entity_json_parse_error",
                error=str(exc),
                raw=raw[:300],
            )
            return EntityMap()

        mentions: list[EntityMention] = []
        for item in data.get("entities", []):
            try:
                mention = EntityMention(
                    canonical=str(item.get("canonical", "")).strip(),
                    aliases=[str(a) for a in item.get("aliases", []) if a],
                    type=str(item.get("type", "OTHER")).upper(),
                    first_mention_offset=int(item.get("first_mention_offset", 0)),
                )
                if mention.canonical:
                    mentions.append(mention)
            except Exception as exc:
                logger.warning(
                    "coref_service.entity_item_parse_error",
                    item=item,
                    error=str(exc),
                )

        return EntityMap(entities=mentions)

    # ── Pass 2: reference resolution ──────────────────────────────────────────

    async def _resolve_references(self, text: str, entity_map: EntityMap) -> str:
        """
        Pass 2 — replace pronouns and ambiguous references with canonical names.

        If Gemini fails or the entity map is empty, the original text is
        returned unchanged.

        Args:
            text:       Full document text.
            entity_map: EntityMap from Pass 1.

        Returns:
            Coreference-resolved text string.
        """
        if not entity_map.entities:
            await logger.adebug(
                "coref_service.skip_resolution_empty_map"
            )
            return text

        entity_map_json = json.dumps(entity_map.model_dump(), ensure_ascii=False)
        prompt = _COREF_RESOLUTION_PROMPT.format(
            entity_map_json=entity_map_json,
            text=text,
        )

        try:
            response = await asyncio.to_thread(self._model.generate_content, prompt)
            resolved = response.text.strip()
            if not resolved:
                await logger.awarning("coref_service.empty_resolved_text")
                return text
            resolved = self._restore_corrupted_parentheticals(text, resolved)
            await logger.ainfo(
                "coref_service.resolution_complete",
                original_chars=len(text),
                resolved_chars=len(resolved),
            )
            return resolved
        except Exception as exc:
            await logger.aerror(
                "coref_service.resolution_failed",
                error=str(exc),
            )
            return text

    # ── Post-processing ───────────────────────────────────────────────────────

    # Matches "ACRONYM (ACRONYM)" — i.e. Gemini replaced an expansion with the
    # same canonical name it already appears as: "OASIS (OASIS)".
    _SELF_REF_PAREN = re.compile(r'\b([A-Z][A-Z0-9]{1,})\s*\(\1\)')

    # Matches the original form in raw text: "ACRONYM (any expansion at least 4 chars)"
    _ORIGINAL_PAREN = re.compile(r'\b([A-Z][A-Z0-9]{1,})\s*\(([^)]{4,})\)')

    def _restore_corrupted_parentheticals(self, original: str, resolved: str) -> str:
        """
        Safety net for Option-C post-processing.

        Detects any ``ACRONYM (ACRONYM)`` patterns that Gemini introduced in the
        resolved text (meaning it replaced a definitional expansion with the
        canonical name), and restores the original ``ACRONYM (expansion)`` text
        found in the raw OCR.

        Args:
            original: Raw OCR text before coref resolution.
            resolved: Gemini-resolved text that may contain corrupted spans.

        Returns:
            Resolved text with definitional parentheticals restored.
        """
        # Build a lookup: acronym → original expansion from raw text
        original_expansions: dict[str, str] = {}
        for match in self._ORIGINAL_PAREN.finditer(original):
            acronym, expansion = match.group(1), match.group(2)
            # Only store if the expansion is actually different from the acronym
            if expansion.strip().upper() != acronym:
                original_expansions[acronym] = expansion

        if not original_expansions:
            return resolved

        restored = resolved
        corrupted_found = 0

        def _fix(m: re.Match) -> str:
            nonlocal corrupted_found
            acronym = m.group(1)
            if acronym in original_expansions:
                corrupted_found += 1
                return f"{acronym} ({original_expansions[acronym]})"
            return m.group(0)

        restored = self._SELF_REF_PAREN.sub(_fix, restored)

        if corrupted_found:
            logger.warning(
                "coref_service.restored_corrupted_parentheticals",
                count=corrupted_found,
            )

        return restored
