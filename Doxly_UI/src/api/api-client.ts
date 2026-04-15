/**
 * api-client.ts
 * =============
 * Type-safe API client for the Document Intelligence backend.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Enums
// ─────────────────────────────────────────────────────────────────────────────

export type DocType =
  | "invoice"
  | "contract"
  | "letter"
  | "form"
  | "note"
  | "report"
  | "receipt"
  | "other";

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
  doc_type: DocType;
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

export interface ErrorResponse {
  error: string;
  message: string;
  detail?: unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// Request types
// ─────────────────────────────────────────────────────────────────────────────

export interface QueryFilters {
  doc_type?: DocType;
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
  public readonly status: number;
  public readonly body: ErrorResponse;

  constructor(
    status: number,
    body: ErrorResponse
  ) {
    super(`API ${status}: ${body.message}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
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
    health: {
      live(): Promise<LivenessResponse> {
        return request(base, "/health/live");
      },
      ready(): Promise<ReadinessResponse> {
        return request(base, "/health/ready");
      },
    },

    auth: {
      loginUrl(userId: string): string {
        return `${base}/auth/login?user_id=${encodeURIComponent(userId)}`;
      },
      status(userId: string): Promise<AuthStatusResponse> {
        return request(base, `/auth/status?user_id=${encodeURIComponent(userId)}`);
      },
      revoke(userId: string): Promise<AuthRevokeResponse> {
        return request(base, `/auth/revoke?user_id=${encodeURIComponent(userId)}`, {
          method: "DELETE",
          headers: {},
        });
      },
    },

    ingest: {
      async upload(userId: string, file: File): Promise<UploadResponse> {
        const form = new FormData();
        form.append("user_id", userId);
        form.append("file", file);

        const res = await fetch(`${base}/ingest/upload`, {
          method: "POST",
          body: form,
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

      flush(sessionId: string, userId: string): Promise<FlushResponse> {
        return request(
          base,
          `/ingest/flush/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(userId)}`,
          { method: "POST", headers: {} }
        );
      },
    },

    query: {
      ask(req: QueryRequest): Promise<QueryResponse> {
        return request(base, "/query", {
          method: "POST",
          body: JSON.stringify(req),
        });
      },

      stream(
        req: QueryRequest,
        onToken: (token: string) => void,
        onDone: () => void,
        onError: (error: Error) => void
      ): void {
        const params = new URLSearchParams({
          user_id: req.user_id,
          query: req.query,
        });
        if (req.filters) {
          params.set("filters", JSON.stringify(req.filters));
        }

        const url = `${base}/query/stream?${params.toString()}`;
        const source = new EventSource(url);

        source.onmessage = (event) => {
          if (event.data === "[DONE]") {
            source.close();
            onDone();
          } else {
            onToken(event.data);
          }
        };

        source.onerror = () => {
          source.close();
          onError(new Error("SSE stream error"));
        };
      },
    },
  };
}

const BASE_URL =
  typeof import.meta !== "undefined"
    ? (import.meta as any).env?.VITE_API_BASE_URL ?? "http://localhost:8000"
    : "http://localhost:8000";

export const api = createApiClient(BASE_URL);
