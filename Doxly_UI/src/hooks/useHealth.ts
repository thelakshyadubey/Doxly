import { useState, useEffect } from "react";
import type { ReadinessResponse } from "../api/api-client";
import { api } from "../api/api-client";

export function useHealth() {
  const [isLive, setIsLive] = useState<boolean | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);

  // Poll liveness every 5 seconds
  useEffect(() => {
    let mounted = true;
    const checkLive = async () => {
      try {
        await api.health.live();
        if (mounted) setIsLive(true);
      } catch {
        if (mounted) setIsLive(false);
      }
    };

    checkLive();
    const interval = setInterval(checkLive, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  // Poll readiness every 30 seconds
  useEffect(() => {
    let mounted = true;
    const checkReady = async () => {
      try {
        const res = await api.health.ready();
        if (mounted) setReadiness(res);
      } catch {
        if (mounted) {
          setReadiness({
            status: "degraded",
            qdrant: false,
            neo4j: false,
            redis: false,
            details: { _error: "Server off" },
          });
        }
      }
    };

    checkReady();
    const interval = setInterval(checkReady, 30000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return { isLive, readiness };
}
