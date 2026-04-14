"""
app/stores/neo4j_store.py
=========================
Neo4j (AuraDB) graph store wrapper.

Uses the official ``neo4j`` driver with AsyncGraphDatabase for true async
operation.  All schema constraints are created idempotently at startup.

Graph schema
------------
(:User {user_id})
  -[:OWNS]->
(:Session {session_id, doc_type, created_at, drive_folder_path})
  -[:CONTAINS]->
(:Chunk {chunk_id, chunk_index, page_num, text_preview, role})
  -[:MENTIONS]->
(:Entity {canonical, type})

(:Entity)-[:ALIAS_OF]->(:Entity)   # alias → canonical

"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.models.domain import Chunk, EntityMap, EntityMention
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum characters stored as text_preview on Chunk nodes
_PREVIEW_LEN = 200


class Neo4jStore:
    """
    Async Neo4j graph store for document knowledge graph construction.

    Args:
        uri:      Neo4j Bolt / AuraDB connection URI.
        user:     Database username.
        password: Database password.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver: Optional[AsyncDriver] = None

    # ── Connection ────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Open the async Neo4j driver.

        Must be called during app lifespan startup before any other method.
        """
        self._driver = AsyncGraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        await logger.ainfo("neo4j_store.connected", uri=self._uri)

    async def close(self) -> None:
        """Close the driver and release all connections."""
        if self._driver:
            await self._driver.close()
            await logger.ainfo("neo4j_store.closed")

    async def ping(self) -> bool:
        """
        Verify connectivity by running a trivial Cypher query.

        Returns:
            True if the database is reachable, False otherwise.
        """
        try:
            async with self._driver.session() as session:
                await session.run("RETURN 1")
            return True
        except Exception as exc:
            await logger.awarning("neo4j_store.ping_failed", error=str(exc))
            return False

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create uniqueness constraints idempotently.

        Safe to call on every startup — the ``IF NOT EXISTS`` clause makes
        all statements no-ops when constraints already exist.
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical IS UNIQUE",
        ]
        async with self._driver.session() as session:
            for stmt in constraints:
                await session.run(stmt)

        await logger.ainfo("neo4j_store.schema_initialized")

    # ── Ingestion writes ──────────────────────────────────────────────────────

    async def ensure_user(self, user_id: str) -> None:
        """
        MERGE a User node (idempotent).

        Args:
            user_id: Opaque user identifier.
        """
        async with self._driver.session() as session:
            await session.run(
                "MERGE (:User {user_id: $user_id})",
                user_id=user_id,
            )

    async def create_session_node(
        self,
        user_id: str,
        session_id: str,
        doc_type: str,
        drive_folder_path: str,
    ) -> None:
        """
        MERGE a Session node and link it to its User.

        Args:
            user_id:           Owning user.
            session_id:        Session UUID.
            doc_type:          Classified document type string.
            drive_folder_path: Drive folder URL/path for this session.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u:User {user_id: $user_id})
                MERGE (s:Session {session_id: $session_id})
                SET s.doc_type = $doc_type,
                    s.created_at = $created_at,
                    s.drive_folder_path = $drive_folder_path
                MERGE (u)-[:OWNS]->(s)
                """,
                user_id=user_id,
                session_id=session_id,
                doc_type=doc_type,
                created_at=created_at,
                drive_folder_path=drive_folder_path,
            )

    async def upsert_chunk(self, chunk: Chunk) -> None:
        """
        MERGE a Chunk node and link it to its parent Session.

        Args:
            chunk: Domain ``Chunk`` object.
        """
        text_preview = chunk.text[:_PREVIEW_LEN]
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.chunk_index = $chunk_index,
                    c.page_num = $page_num,
                    c.text_preview = $text_preview,
                    c.role = $role
                WITH c
                MATCH (s:Session {session_id: $session_id})
                MERGE (s)-[:CONTAINS]->(c)
                """,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_num=chunk.page_num,
                text_preview=text_preview,
                role=chunk.role.value,
                session_id=chunk.session_id,
            )

    async def upsert_entities(
        self, chunk_id: str, entity_map: EntityMap
    ) -> None:
        """
        MERGE Entity nodes and link them to the Chunk via MENTIONS.

        Also creates ALIAS_OF relationships between surface-form aliases and
        the canonical entity node.

        Args:
            chunk_id:   UUID of the chunk that mentions these entities.
            entity_map: Extracted entity data.
        """
        async with self._driver.session() as session:
            for entity in entity_map.entities:
                # Create canonical entity node and connect to chunk
                await session.run(
                    """
                    MERGE (e:Entity {canonical: $canonical})
                    SET e.type = $type
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    canonical=entity.canonical,
                    type=entity.type,
                    chunk_id=chunk_id,
                )

                # Create alias nodes that point back to the canonical
                for alias in entity.aliases:
                    if alias and alias != entity.canonical:
                        await session.run(
                            """
                            MERGE (a:Entity {canonical: $alias})
                            MERGE (c:Entity {canonical: $canonical})
                            MERGE (a)-[:ALIAS_OF]->(c)
                            """,
                            alias=alias,
                            canonical=entity.canonical,
                        )

    # ── Retrieval queries ─────────────────────────────────────────────────────

    async def get_sibling_chunks(self, chunk_id: str) -> list[str]:
        """
        Return chunk_ids of all chunks in the same session as *chunk_id*.

        Args:
            chunk_id: UUID of any chunk within the target session.

        Returns:
            List of chunk_id strings (excluding the input chunk).
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})<-[:CONTAINS]-(s:Session)-[:CONTAINS]->(sibling:Chunk)
                WHERE sibling.chunk_id <> $chunk_id
                RETURN sibling.chunk_id AS chunk_id
                """,
                chunk_id=chunk_id,
            )
            return [record["chunk_id"] async for record in result]

    async def get_chunks_by_entity(
        self, canonical_name: str, user_id: str
    ) -> list[str]:
        """
        Return chunk_ids of all chunks that mention a given entity,
        scoped to the requesting user.

        Args:
            canonical_name: Canonical entity name to look up.
            user_id:        Tenant filter — only chunks owned by this user.

        Returns:
            List of chunk_id strings.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User {user_id: $user_id})-[:OWNS]->(s:Session)-[:CONTAINS]->(c:Chunk)
                      -[:MENTIONS]->(e:Entity {canonical: $canonical})
                RETURN c.chunk_id AS chunk_id
                """,
                canonical=canonical_name,
                user_id=user_id,
            )
            return [record["chunk_id"] async for record in result]

    async def get_entity_subgraph(self, session_id: str) -> list[dict]:
        """
        Return the full entity relationship graph for a session.

        Args:
            session_id: Session UUID.

        Returns:
            List of dicts with keys ``entity``, ``type``, ``chunk_count``.
        """
        async with self._driver.session() as session:
            result = await session.run(
                """
                MATCH (s:Session {session_id: $session_id})-[:CONTAINS]->(c:Chunk)
                      -[:MENTIONS]->(e:Entity)
                RETURN e.canonical AS entity, e.type AS type, count(c) AS chunk_count
                ORDER BY chunk_count DESC
                """,
                session_id=session_id,
            )
            return [
                {
                    "entity": record["entity"],
                    "type": record["type"],
                    "chunk_count": record["chunk_count"],
                }
                async for record in result
            ]
