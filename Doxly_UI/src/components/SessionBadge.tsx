import React from "react";
import { clsx } from "clsx";
import type { SessionStatus } from "../api/api-client";

export function SessionBadge({ status }: { status: SessionStatus | null }) {
  if (!status) return null;

  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold capitalize",
        status === "open" && "bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-300",
        status === "queued" && "bg-yellow-100 text-yellow-800 dark:bg-yellow-500/20 dark:text-yellow-300",
        status === "processing" && "bg-orange-100 text-orange-800 dark:bg-orange-500/20 dark:text-orange-300",
        status === "indexed" && "bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-300",
        status === "failed" && "bg-red-100 text-red-800 dark:bg-red-500/20 dark:text-red-300"
      )}
    >
      {status === "processing" && (
        <svg className="animate-spin -ml-1 mr-1.5 h-3 w-3 " xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {status}
    </span>
  );
}
