import React from "react";
import { ServerCrash } from "lucide-react";

export function ServerOfflineBanner() {
  return (
    <div className="fixed inset-0 z-[100] bg-zinc-50/90 dark:bg-zinc-950/90 backdrop-blur-md flex items-center justify-center animate-in fade-in duration-300">
      <div className="flex flex-col items-center text-center p-8 max-w-sm">
        <div className="rounded-full bg-red-100 dark:bg-red-500/20 p-4 mb-4">
          <ServerCrash className="w-12 h-12 text-red-600 dark:text-red-400" />
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 mb-2">
          Server Offline
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          We cannot reach the application server. The interface is blocked. 
          Will automatically reconnect when the server is healthy.
        </p>
        <div className="mt-8 flex items-center gap-2 text-xs font-medium text-zinc-400 dark:text-zinc-500 uppercase tracking-widest">
          <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></span>
          Attempting to reconnect
        </div>
      </div>
    </div>
  );
}
