import React from "react";
import ReactMarkdown from "react-markdown";
import { Copy, FileText, Check as CheckIcon } from "lucide-react";
import type { Citation } from "../api/api-client";

export function AnswerDisplay({
  answer,
  citations = [],
  isAnswering,
  isComplete
}: {
  answer: string;
  citations?: Citation[];
  isAnswering: boolean;
  isComplete: boolean;
}) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(answer).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!answer && !isAnswering && !isComplete) return null;

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

        <div className="prose prose-sm dark:prose-invert max-w-none leading-relaxed mb-6">
          <ReactMarkdown>{answer}</ReactMarkdown>
          {isAnswering && <span className="inline-block w-2 text-primary animate-pulse ml-1">▍</span>}
        </div>

      </div>

      {isComplete && citations.length > 0 && (
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
