/**
 * api-client.ts
 * =============
 * Type-safe API client for the Document Intelligence backend.
 *
 * Usage:
 *   import { createApiClient } from "./api-client";
 *   const api = createApiClient("http://localhost:8000");
 *
 *   await api.auth.status("alice");
 *   await api.ingest.upload("alice", file);
 *   await api.query.ask({ user_id: "alice", query: "What is the total?" });
 */

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

export type SessionStatus =
  | "open"
  | "queued"
  | "processing"
  | "indexed"
  | "failed";

// ─────────────────────────────────────────────────────────────────────────────
// Response types
// ─────────────────────────────────────────────────────────────────────────────

export interface LivenessResponse {
  status: "ok";
}

export interface ReadinessResponse {
  status: "ok" | "degraded";
  qdrant: boolean;
  neo4j: boolean;
  redis: boolean;
  details: Record<string, string>;
}

export interface AuthStatusResponse {
  user_id: string;
  authorized: boolean;
}

export interface AuthCallbackResponse {
  authorized: boolean;
  user_id: string;
  message: string;
}

export interface AuthRevokeResponse {
  user_id: string;
  authorized: false;
  message: string;
}

export interface UploadResponse {
  session_id: string;
  page_count: number;
  status: SessionStatus;
}

export interface FlushResponse {
  session_id: string;
  chunk_count: number;
  entity_count: number;
  status: SessionStatus;
}

export interface Citation {
  source: string;
  page: number;
  chunk_id: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  confidence: number;
}

export interface GraphNode {
  id: string;
  label: string;
  node_type: "session" | "chunk" | "entity";
  page_num?: number;
  text_preview?: string;
  role?: string;
  entity_type?: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ErrorResponse {
  error: string;
  message: string;
  detail?: unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// Request types
// ─────────────────────────────────────────────────────────────────────────────

export interface QueryFilters {
  session_id?: string;
}

export interface QueryRequest {
  user_id: string;
  query: string;
  filters?: QueryFilters;
}

// ─────────────────────────────────────────────────────────────────────────────
// Error class
// ─────────────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: ErrorResponse
  ) {
    super(`API ${status}: ${body.message}`);
    this.name = "ApiError";
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal fetch helper
// ─────────────────────────────────────────────────────────────────────────────

async function request<T>(
  baseUrl: string,
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    headers: { "Content-Type": "application/json", ...init.headers },
    ...init,
  });

  if (!res.ok) {
    let body: ErrorResponse;
    try {
      body = await res.json();
    } catch {
      body = { error: "unknown", message: res.statusText };
    }
    throw new ApiError(res.status, body);
  }

  return res.json() as Promise<T>;
}

// ─────────────────────────────────────────────────────────────────────────────
// API client factory
// ─────────────────────────────────────────────────────────────────────────────

export function createApiClient(baseUrl: string = "http://localhost:8000") {
  // Strip trailing slash
  const base = baseUrl.replace(/\/$/, "");

  return {
    // ── Health ──────────────────────────────────────────────────────────────

    health: {
      /** Always returns { status: "ok" } if the server is running. */
      live(): Promise<LivenessResponse> {
        return request(base, "/health/live");
      },

      /** Returns connectivity status for Qdrant, Neo4j, and Redis. */
      ready(): Promise<ReadinessResponse> {
        return request(base, "/health/ready");
      },
    },

    // ── Auth ─────────────────────────────────────────────────────────────────

    auth: {
      /**
       * Returns the Google OAuth login URL to redirect the user to.
       * The URL is: `{base}/auth/login?user_id={userId}`
       * Redirect the browser to this URL — do not fetch it.
       */
      loginUrl(userId: string): string {
        return `${base}/auth/login?user_id=${encodeURIComponent(userId)}`;
      },

      /**
       * Check if a user has already completed Google OAuth.
       * Call this on app load to skip re-auth if already authorized.
       */
      status(userId: string): Promise<AuthStatusResponse> {
        return request(base, `/auth/status?user_id=${encodeURIComponent(userId)}`);
      },

      /**
       * Revoke a user's Drive authorization (deletes tokens from Redis).
       */
      revoke(userId: string): Promise<AuthRevokeResponse> {
        return request(base, `/auth/revoke?user_id=${encodeURIComponent(userId)}`, {
          method: "DELETE",
          headers: {},
        });
      },
    },

    // ── Ingest ───────────────────────────────────────────────────────────────

    ingest: {
      /**
       * Upload a single image page for OCR.
       * Returns a session_id — reuse it for all pages of the same document.
       * Call flush() when all pages are uploaded.
       *
       * @param userId  - The user identifier
       * @param file    - Image file (JPEG / PNG / TIFF / PDF page)
       */
      async upload(userId: string, file: File): Promise<UploadResponse> {
        const form = new FormData();
        form.append("user_id", userId);
        form.append("file", file);

        const res = await fetch(`${base}/ingest/upload`, {
          method: "POST",
          body: form,
          // Do NOT set Content-Type here — browser sets multipart boundary automatically
        });

        if (!res.ok) {
          let body: ErrorResponse;
          try {
            body = await res.json();
          } catch {
            body = { error: "unknown", message: res.statusText };
          }
          throw new ApiError(res.status, body);
        }

        return res.json() as Promise<UploadResponse>;
      },

      /**
       * Trigger the full ingestion pipeline for a session.
       * Blocks until classify → coref → chunk → embed → store completes.
       * Safe to call multiple times (idempotent if already indexed).
       *
       * @param sessionId - From the UploadResponse
       * @param userId    - Must match the uploading user
       */
      flush(sessionId: string, userId: string): Promise<FlushResponse> {
        return request(
          base,
          `/ingest/flush/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
          { method: "POST", headers: {} }
        );
      },
    },

    // ── Query ────────────────────────────────────────────────────────────────

    query: {
      /**
       * Ask a question. Returns full answer + citations + confidence.
       * Waits for the complete response before resolving.
       *
       * @param req - { user_id, query, filters? }
       */
      ask(req: QueryRequest): Promise<QueryResponse> {
        return request(base, "/query", {
          method: "POST",
          body: JSON.stringify(req),
        });
      },

      /**
       * Fetch the Neo4j reasoning subgraph for the given chunk IDs.
       * Pass the chunk_ids from a QueryResponse's citations array.
       *
       * @param userId   - Owning user
       * @param chunkIds - chunk_id values from Citation objects
       */
      graph(userId: string, chunkIds: string[]): Promise<GraphResponse> {
        const params = new URLSearchParams({ user_id: userId });
        chunkIds.forEach((id) => params.append("chunk_ids", id));
        return request(base, `/query/graph?${params.toString()}`);
      },
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Default singleton (uses VITE env var if available, else localhost)
// ─────────────────────────────────────────────────────────────────────────────

const BASE_URL =
  typeof import.meta !== "undefined"
    ? (import.meta as any).env?.VITE_API_BASE_URL ?? "http://localhost:8000"
    : "http://localhost:8000";

export const api = createApiClient(BASE_URL);
