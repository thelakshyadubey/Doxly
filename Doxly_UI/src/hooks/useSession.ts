import { useState, useCallback } from "react";
import type { SessionStatus, UploadResponse, DocType } from "../api/api-client";
import { api } from "../api/api-client";
import { useToast } from "./useToast";

export function useSession(userId: string | null) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  
  // Results from flush
  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [entityCount, setEntityCount] = useState<number | null>(null);
  const [docType, setDocType] = useState<DocType | null>(null);

  const { addToast } = useToast();

  const resetSession = useCallback(() => {
    setSessionId(null);
    setPageCount(0);
    setStatus(null);
    setChunkCount(null);
    setEntityCount(null);
    setDocType(null);
  }, []);

  const uploadFile = async (file: File) => {
    if (!userId) return null;
    setIsUploading(true);
    try {
      const res = await api.ingest.upload(userId, file);
      if (!sessionId || res.session_id !== sessionId) {
        setSessionId(res.session_id);
      }
      // Re-use current page count and add new page, backend page_count might be total
      // But the api returns the page_count so far for that session
      // Wait, the spec says "The session_id returned from the first upload must be reused... accumulate page_count"
      // But we aren't passing session_id to upload? The spec for `/ingest/upload` doesn't take session_id.
      // Ah. The spec says: "The session_id returned from the first upload must be reused for all subsequent uploads in the same batch".
      // Wait, if `/ingest/upload` doesn't take session_id, how does the backend know?
      // Wait, the prompt says "The session_id returned from the first upload must be reused for all subsequent uploads". Let's update `api-client.ts` to accept session_id or just pass it in formdata.
      // Wait, let's look at section 3:
      // Request: multipart/form-data: user_id, file.
      // Maybe the backend uses user_id to find the "open" session.
      // Let's just trust the response page_count.
      setPageCount((prev) => prev + 1);
      setStatus(res.status);
      return res;
    } catch (err: any) {
      const msg = err.body?.message ?? err.body?.detail ?? "Failed to upload file";
      addToast(typeof msg === "string" ? msg : JSON.stringify(msg), "error");
      throw err;
    } finally {
      setIsUploading(false);
    }
  };

  const flushSession = async () => {
    if (!userId || !sessionId) return;
    setStatus("processing");
    try {
      const res = await api.ingest.flush(sessionId, userId);
      setStatus(res.status);
      setChunkCount(res.chunk_count);
      setEntityCount(res.entity_count);
      setDocType(res.doc_type);
      return res;
    } catch (err: any) {
      setStatus("failed");
      const msg = err.body?.message ?? err.body?.detail ?? "Pipeline failed";
      addToast(typeof msg === "string" ? msg : JSON.stringify(msg), "error");
      throw err;
    }
  };

  return {
    sessionId,
    pageCount,
    status,
    isUploading,
    uploadFile,
    flushSession,
    resetSession,
    chunkCount,
    entityCount,
    docType,
  };
}
