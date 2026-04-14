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
  → OCRService (Google Vision DOCUMENT_TEXT_DETECTION)
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
| Async strategy | All I/O is async; synchronous clients (Qdrant, Vision, Drive) are wrapped in `asyncio.to_thread`. |
| Embedding | `text-embedding-004`, 768-dim, `retrieval_document` for chunks / `retrieval_query` for queries. |

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

### Singletons on `app.state`

`app/main.py` lifespan attaches: `qdrant_store`, `neo4j_store`, `redis_store`, `gemini_model`, `gemini_embed_model`, `ocr_service`, `drive_service`. Routes pull them via `app/api/dependencies.py`.

### Logging

`app/utils/logger.py` — structlog JSON pipeline. Every module: `logger = get_logger(__name__)`.
