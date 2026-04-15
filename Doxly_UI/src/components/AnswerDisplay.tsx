import React from "react";
import { Copy, FileText, Check as CheckIcon } from "lucide-react";
import type { Citation } from "../api/api-client";
import { clsx } from "clsx";

export function AnswerDisplay({
  answer,
  citations = [],
  confidence = 0,
  isStreaming,
  isComplete
}: {
  answer: string;
  citations?: Citation[];
  confidence?: number;
  isStreaming: boolean;
  isComplete: boolean;
}) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(answer).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!answer && !isStreaming && !isComplete) return null;

  return (
    <div className="flex flex-col gap-6 animate-in fade-in duration-300">
      <div className="glass-card p-6 relative group">
        <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
           <button 
             onClick={handleCopy}
             className="p-1.5 rounded-md hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-500 transition-colors"
           >
             {copied ? <CheckIcon className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
           </button>
        </div>
        
        <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed mb-6 whitespace-pre-wrap">
          {answer}
          {isStreaming && <span className="inline-block w-2 text-primary animate-pulse ml-1">▍</span>}
        </div>

        {isComplete && confidence !== 0 && (
          <div className="flex items-center gap-3 pt-6 border-t border-zinc-100 dark:border-zinc-800">
            <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500">Confidence</span>
            <div className="flex-1 max-w-[200px] h-2 bg-zinc-100 dark:bg-zinc-800 rounded-full overflow-hidden flex">
              <div 
                className={clsx(
                  "h-full transition-all duration-1000",
                  confidence > 0.7 ? "bg-emerald-500" : confidence > 0.4 ? "bg-yellow-500" : "bg-red-500"
                )} 
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className="text-xs font-mono">{Math.round(confidence * 100)}%</span>
          </div>
        )}
      </div>

      {isComplete && citations && citations.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" /> Citations
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {citations.map((c, i) => (
              <div key={i} className="glass-card p-3 flex flex-col gap-1 text-xs group cursor-default">
                <span className="font-mono text-[10px] text-zinc-500 flex items-center justify-between">
                  ID: {c.chunk_id.substring(0, 8)}...
                  <span className="bg-primary/10 text-primary px-1.5 py-0.5 rounded font-bold">Page {c.page}</span>
                </span>
                <span className="truncate font-medium text-zinc-700 dark:text-zinc-300 mt-1" title={c.source}>
                  {c.source.split('/').pop()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
