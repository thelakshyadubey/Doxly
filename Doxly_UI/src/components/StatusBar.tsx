import React from "react";
import { useHealth } from "../hooks/useHealth";
import { clsx } from "clsx";
import { Server, Database, Box } from "lucide-react";

export function StatusBar() {
  const { readiness } = useHealth();

  if (!readiness) {
    return (
      <div className="fixed bottom-0 left-0 w-full bg-zinc-900 border-t border-zinc-800 p-2 z-50 flex items-center justify-center text-xs text-zinc-400">
        Checking systems...
      </div>
    );
  }

  const { status, qdrant, neo4j, redis } = readiness;

  return (
    <div
      className={clsx(
        "fixed bottom-0 left-0 w-full z-50 transition-colors duration-300 border-t flex flex-col",
        status === "ok"
          ? "bg-white/80 dark:bg-zinc-950/80 backdrop-blur border-zinc-200 dark:border-zinc-800"
          : "bg-yellow-50 dark:bg-yellow-900/30 border-yellow-200 dark:border-yellow-700 backdrop-blur"
      )}
    >
      {status === "degraded" && (
        <div className="w-full bg-yellow-100 dark:bg-yellow-900/50 text-yellow-800 dark:text-yellow-200 text-xs font-semibold text-center py-1 border-b border-yellow-200 dark:border-yellow-800 transition-all">
          Services degraded. Functionality may be limited.
        </div>
      )}
      <div className="flex items-center justify-between px-4 py-2 text-xs font-medium text-zinc-600 dark:text-zinc-400 max-w-7xl mx-auto w-full">
        <div className="flex items-center gap-1.5 opacity-70">
          <Server className="w-3.5 h-3.5" /> Core API
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "w-2 h-2 rounded-full",
                qdrant ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"
              )}
            />
            <span className="flex items-center gap-1">
              <Box className="w-3.5 h-3.5 opacity-60" /> Qdrant
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "w-2 h-2 rounded-full",
                neo4j ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"
              )}
            />
            <span className="flex items-center gap-1">
              <Database className="w-3.5 h-3.5 opacity-60" /> Neo4j
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                "w-2 h-2 rounded-full",
                redis ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" : "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]"
              )}
            />
            <span className="flex items-center gap-1">
              <Database className="w-3.5 h-3.5 opacity-60" /> Redis
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
