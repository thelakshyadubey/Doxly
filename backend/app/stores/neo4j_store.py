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

from backend.app.models.domain import Chunk, EntityMap, EntityMention
from backend.app.utils.logger import get_logger

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
        drive_folder_path: str,
    ) -> None:
        """
        MERGE a Session node and link it to its User.

        Args:
            user_id:           Owning user.
            session_id:        Session UUID.
            drive_folder_path: Drive folder URL/path for this session.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        async with self._driver.session() as session:
            await session.run(
                """
                MERGE (u:User {user_id: $user_id})
                MERGE (s:Session {session_id: $session_id})
                SET s.created_at = $created_at,
                    s.drive_folder_path = $drive_folder_path
                MERGE (u)-[:OWNS]->(s)
                """,
                user_id=user_id,
                session_id=session_id,
                created_at=created_at,
                drive_folder_path=drive_folder_path,
            )

    async def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """
        MERGE all Chunk nodes and their Session links in a single UNWIND query.

        Args:
            chunks: List of domain ``Chunk`` objects to upsert.
        """
        if not chunks:
            return
        rows = [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "page_num": c.page_num,
                "text_preview": c.text[:_PREVIEW_LEN],
                "role": c.role.value,
                "session_id": c.session_id,
            }
            for c in chunks
        ]
        async with self._driver.session() as session:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.chunk_index  = row.chunk_index,
                    c.page_num     = row.page_num,
                    c.text_preview = row.text_preview,
                    c.role         = row.role
                WITH c, row
                MATCH (s:Session {session_id: row.session_id})
                MERGE (s)-[:CONTAINS]->(c)
                """,
                rows=rows,
            )

    async def upsert_entities_batch(
        self, chunk_entity_pairs: list[tuple[str, EntityMap]]
    ) -> None:
        """
        MERGE all Entity nodes, MENTIONS links, and ALIAS_OF relationships in
        two UNWIND queries (one for entity nodes + mentions, one for aliases).

        Args:
            chunk_entity_pairs: List of (chunk_id, EntityMap) tuples.
        """
        entity_rows: list[dict] = []
        alias_rows: list[dict] = []

        for chunk_id, entity_map in chunk_entity_pairs:
            for entity in entity_map.entities:
                entity_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "canonical": entity.canonical,
                        "type": entity.type,
                    }
                )
                for alias in entity.aliases:
                    if alias and alias != entity.canonical:
                        alias_rows.append(
                            {"alias": alias, "canonical": entity.canonical}
                        )

        if not entity_rows:
            return

        async with self._driver.session() as session:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {canonical: row.canonical})
                SET e.type = row.type
                WITH e, row
                MATCH (c:Chunk {chunk_id: row.chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                rows=entity_rows,
            )
            if alias_rows:
                await session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (a:Entity {canonical: row.alias})
                    MERGE (cn:Entity {canonical: row.canonical})
                    MERGE (a)-[:ALIAS_OF]->(cn)
                    """,
                    rows=alias_rows,
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
