# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.11 (`.venv` in project root)
- Activate before running anything:
  ```bash
  source .venv/Scripts/activate   # Git Bash / bash on Windows
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Copy and fill in secrets:
  ```bash
  cp .env.example .env
  ```

## Running the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs auto-generated at `http://localhost:8000/docs`.

## Architecture

**Document Intelligence pipeline** — upload images → OCR → classify → coref resolve → chunk + embed → Qdrant + Neo4j → hybrid QnA. Each user authorizes their own Google Drive via OAuth 2.0; documents are stored in the user's own Drive, not a shared service-account folder.

### Request flow

```
GET /auth/login?user_id={user_id}
  → AuthService.get_authorization_url()   (state stored in Redis, TTL=600s)
  → 302 redirect to Google consent screen

GET /auth/callback?code=...&state=...
  → AuthService.exchange_code()           (state consumed from Redis — replay-safe)
  → tokens persisted to Redis: tokens:{user_id}

POST /ingest/upload
  → load tokens:{user_id} from Redis → DriveService.for_user(tokens) → UserDriveClient
  → OCRService (Gemini multimodal — image bytes → JSON {text, language})
  → SessionService (Redis window: floor(ts / threshold) bucket)
  → UserDriveClient.create_session_folder(session_id, "pending")
  → UserDriveClient.upload_bytes()        (original image → user's own Drive)
  → save refreshed tokens back to Redis

POST /ingest/flush/{session_id}
  → load tokens:{user_id} from Redis → UserDriveClient
  → IngestionOrchestrator coordinates:
      ClassificationService  (Gemini, first 500 chars → doc_type)
      UserDriveClient        (create folder My Drive/{app_folder}/{session_id}/{doc_type}/)
      CorefService           (2-pass Gemini: entity extract → resolve pronouns)
      ChunkingService        (page=parent, sliding-window children, Gemini embeddings)
      QdrantStore.upsert     (batched, with entity header + full payload)
      Neo4jStore writes      (User→Session→Chunk→Entity graph)
  → save refreshed tokens back to Redis

POST /query  /  GET /query/stream
  → RetrievalService:
      asyncio.gather(Qdrant vector search, Neo4j entity lookup, parent chunk fetch)
      RRF fusion (k=60)
  → AnswerService (Gemini with context block, streaming or non-streaming)
```

### Key design decisions

| Concern | Decision |
|---|---|
| Per-user Drive | `DriveService` on `app.state` is a factory (holds OAuth client config only). Each request calls `drive_service.for_user(tokens)` to get a `UserDriveClient` scoped to that user's Drive. |
| Token refresh | `google-auth` refreshes the access token transparently. After every Drive operation, `user_drive_client.get_current_tokens()` is saved back to Redis so the updated token survives restarts. |
| OAuth CSRF | Random `state` stored in Redis (`oauth_state:{state}`, TTL=600s). `consume_oauth_state` reads and deletes atomically — prevents replay. |
| Drive folder path | `My Drive / {GOOGLE_DRIVE_APP_FOLDER_NAME} / {session_id} / {doc_type} /` — no `user_id` in path since each user's Drive is already isolated. |
| Session windowing | `session_id = UUID5("session" \| user_id \| str(floor(unix_ts / threshold)))` — all uploads in the same time bucket share a session. |
| Tenant isolation | Every Qdrant search has a mandatory `user_id` filter. |
| Chunk IDs | Deterministic UUID5 from `(session_id, page_num, chunk_index)` → upserts are idempotent. |
| Qdrant mode | `QDRANT_URL=""` → local persistent; set URL → cloud, zero code change. |
| Async strategy | All I/O is async; synchronous clients (Qdrant, Gemini, Drive) are wrapped in `asyncio.to_thread`. |
| Embedding | `text-embedding-004`, 3072-dim, `retrieval_document` for chunks / `retrieval_query` for queries. |
| OCR engine | `OCRService` uses Gemini multimodal (`glm.Part` + `glm.Blob`). No GCP service account needed. Returns `OCRResult{text, confidence=1.0, page_count=1, language}`. |

### Auth flow (new)

- `app/services/auth_service.py` — `AuthService`: `get_authorization_url(user_id)` and `exchange_code(code, state)` using `google_auth_oauthlib.flow.Flow`.
- `app/api/routes/auth.py` — `GET /auth/login`, `GET /auth/callback`, `GET /auth/status`, `DELETE /auth/revoke`.
- Both ingest endpoints return **HTTP 401** with an `/auth/login` hint if the user has no tokens in Redis.

### Drive service (refactored)

- `DriveService` — factory; holds `client_id`, `client_secret`, `app_folder_name`. No user credentials. Attach to `app.state` once at startup.
- `UserDriveClient` — wraps `google.oauth2.credentials.Credentials` for one user. All Drive methods live here. Call `get_current_tokens()` after operations to capture any refreshed access token.

### Store layer

- `app/stores/qdrant_store.py` — synchronous qdrant-client (to_thread in callers), idempotent collection + payload index init.
- `app/stores/neo4j_store.py` — async neo4j driver, constraints bootstrapped at startup.
- `app/stores/redis_store.py` — redis.asyncio; three key namespaces:
  - `session:{user_id}:{session_id}` — `SessionMetadata` JSON, TTL = `SESSION_THRESHOLD_SECONDS`
  - `tokens:{user_id}` — OAuth token dict (JSON), no TTL
  - `oauth_state:{state}` — CSRF mapping `state → user_id`, TTL = 600 s

### Config

`app/config/settings.py` — single `Settings(BaseSettings)` singleton via `lru_cache`. Every value comes from `.env`. Key properties: `qdrant_is_local` (drives client mode), `oauth_client_config` (returns the dict expected by `google_auth_oauthlib.flow.Flow`).

**Required `.env` fields** (no defaults — app won't start without these):
`GEMINI_API_KEY`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `NEO4J_URI`, `NEO4J_PASSWORD`, `REDIS_URL`

**Removed fields** (no longer needed): `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS` — Vision API is gone.

### Singletons on `app.state`

`app/main.py` lifespan attaches: `qdrant_store`, `neo4j_store`, `redis_store`, `gemini_model`, `gemini_embed_model`, `ocr_service`, `drive_service`. Routes pull them via `app/api/dependencies.py`.

### Logging

`app/utils/logger.py` — structlog JSON pipeline. Every module: `logger = get_logger(__name__)`.

### Frontend API client

`api-client.ts` (project root) — type-safe TypeScript client wrapping all endpoints. Uses `fetch` + `EventSource` (no dependencies). Import and call `createApiClient(baseUrl)` or use the default `api` singleton. Set `VITE_API_BASE_URL` in the frontend `.env`.

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
