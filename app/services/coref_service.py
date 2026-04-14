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

from app.models.domain import EntityMap, EntityMention
from app.utils.logger import get_logger

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
Preserve all original punctuation, line breaks, spacing, and [PAGE_BREAK] markers exactly.
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
