import { useState, useCallback } from "react";
import type { SessionStatus } from "../api/api-client";
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

  const { addToast } = useToast();

  const resetSession = useCallback(() => {
    setSessionId(null);
    setPageCount(0);
    setStatus(null);
    setChunkCount(null);
    setEntityCount(null);
  }, []);

  const uploadFile = async (file: File) => {
    if (!userId) return null;
    setIsUploading(true);
    try {
      const res = await api.ingest.upload(userId, file);
      if (!sessionId || res.session_id !== sessionId) {
        setSessionId(res.session_id);
      }
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
  };
}
