import React, { useState, useRef, useEffect } from "react";
import { Send, Upload, Sparkles, LogOut } from "lucide-react";
import type { Citation } from "../api/api-client";
import { api } from "../api/api-client";
import { AnswerDisplay } from "./AnswerDisplay";
import { useToast } from "../hooks/useToast";

export function QueryScreen({
  userId,
  currentSessionId,
  onLogout,
  onUploadMore
}: {
  userId: string;
  currentSessionId: string | null;
  onLogout: () => void;
  onUploadMore: () => void;
}) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [isAnswering, setIsAnswering] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const { addToast } = useToast();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [query]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!query.trim() || isAnswering) return;

    setIsAnswering(true);
    setIsComplete(false);
    setAnswer("");
    setCitations([]);

    const reqObj: any = { user_id: userId, query };
    if (currentSessionId) reqObj.filters = { session_id: currentSessionId };

    try {
      const res = await api.query.ask(reqObj);
      setAnswer(res.answer);
      setCitations(res.citations);
    } catch (err: any) {
      addToast(err.body?.message ?? "Failed to fetch answer", "error");
    } finally {
      setIsAnswering(false);
      setIsComplete(true);
      setQuery("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col min-h-screen px-4 animate-in fade-in duration-300 max-w-4xl mx-auto w-full pb-24">
      <header className="flex items-center justify-between py-6 mb-4">
        <h1 className="text-xl font-bold tracking-tight">Doxly Query</h1>
        <div className="flex items-center gap-4">
          <button
            onClick={onUploadMore}
            className="flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-primary transition-colors"
          >
            <Upload className="w-4 h-4" /> Upload More
          </button>
          <button
            onClick={onLogout}
            className="flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-red-500 transition-colors"
          >
            <LogOut className="w-4 h-4" /> Revoke
          </button>
        </div>
      </header>

      <div className="flex flex-col flex-1 gap-8">
        <div className="flex-1">
          {(!answer && !isAnswering) ? (
            <div className="h-full flex flex-col items-center justify-center text-center opacity-70">
              <Sparkles className="w-12 h-12 text-primary/40 mb-4" />
              <h2 className="text-xl font-medium mb-2">Ask anything about your documents</h2>
              <p className="text-sm max-w-sm mb-8">
                Get AI-generated answers with precise citations pointing directly to your Google Drive files.
              </p>
            </div>
          ) : (
            <AnswerDisplay
              answer={answer}
              citations={citations}
              isAnswering={isAnswering}
              isComplete={isComplete}
              userId={userId}
            />
          )}
        </div>

        <div className="sticky bottom-12 z-10">
          <div className="glass-card p-2 p-3 bg-zinc-50/95 dark:bg-zinc-900/95 shadow-xl shadow-zinc-200/50 dark:shadow-none">
            <form onSubmit={handleSubmit} className="flex items-end gap-2">
              <div className="flex-1 bg-white dark:bg-zinc-950 rounded-xl border border-zinc-200 dark:border-zinc-800 focus-within:border-primary focus-within:ring-1 focus-within:ring-primary overflow-hidden transition-all">
                <textarea
                  ref={textareaRef}
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question..."
                  className="w-full resize-none bg-transparent p-3 max-h-48 outline-none text-sm leading-relaxed"
                  rows={1}
                />
              </div>
              <button
                type="submit"
                disabled={!query.trim() || isAnswering}
                className="bg-primary hover:bg-primary-dark text-white p-3.5 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all flex-shrink-0"
              >
                {isAnswering
                  ? <div className="w-5 h-5 border-2 border-white/30 border-t-white animate-spin rounded-full" />
                  : <Send className="w-5 h-5 -ml-0.5 mt-0.5" />}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
