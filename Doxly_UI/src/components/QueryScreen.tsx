import React, { useState, useRef, useEffect } from "react";
import { Send, Upload, Settings2, Sparkles, LogOut } from "lucide-react";
import type { DocType, Citation } from "../api/api-client";
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
  const [isStreamingMode, setIsStreamingMode] = useState(true);
  const [docTypeFilter, setDocTypeFilter] = useState<DocType | "all">("all");
  const [sessionFilter, setSessionFilter] = useState(currentSessionId || "");
  const [showFilters, setShowFilters] = useState(false);
  
  const [answer, setAnswer] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [confidence, setConfidence] = useState(0);
  
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
    setConfidence(0);

    const filters: any = {};
    if (docTypeFilter !== "all") filters.doc_type = docTypeFilter;
    if (sessionFilter.trim()) filters.session_id = sessionFilter.trim();

    const reqObj = Object.keys(filters).length > 0 
      ? { user_id: userId, query, filters } 
      : { user_id: userId, query };

    if (isStreamingMode) {
      api.query.stream(
        reqObj,
        (token) => {
          setAnswer((prev) => prev + token);
        },
        () => {
          setIsAnswering(false);
          setIsComplete(true);
          // Note: In SSE stream as per spec, backend returns text. 
          // Will assume citations not in stream unless it's JSON encoded token chunks. 
          // The prompt says "uses POST /query (returns complete answer at once)", 
          // for streaming "tokens appear one by one ... [DONE]". Standard SSE.
          // Confidence/citations might not arrive in purely text stream. Wait, the spec says
          // answer appears. Let's just handle text.
        },
        (error) => {
          setIsAnswering(false);
          addToast("Streaming failed. Showing partial answer.", "error");
        }
      );
    } else {
      try {
        const res = await api.query.ask(reqObj);
        setAnswer(res.answer);
        setCitations(res.citations);
        setConfidence(res.confidence);
      } catch (err: any) {
        if (err.body?.message) {
          addToast(err.body.message, "error");
        } else {
          addToast("Failed to fetch answer", "error");
        }
      } finally {
        setIsAnswering(false);
        setIsComplete(true);
      }
    }
    
    // Clear input
    setQuery("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // The 8 DOC types
  const DOC_TYPES: DocType[] = ["invoice", "contract", "letter", "form", "note", "report", "receipt", "other"];

  return (
    <div className="flex flex-col min-h-screen px-4 animate-in fade-in duration-300 max-w-4xl mx-auto w-full pb-24">
      <header className="flex items-center justify-between py-6 mb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Doxly Query</h1>
        </div>
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
               <h2 className="text-xl font-medium mb-2">Ask anything — about your documents or anything else</h2>
               <p className="text-sm max-w-sm mb-8">
                 Get AI-generated answers with precise citations pointing directly to your Google Drive files.
               </p>
            </div>
          ) : (
            <AnswerDisplay 
              answer={answer} 
              citations={citations} 
              confidence={confidence} 
              isStreaming={isAnswering && isStreamingMode} 
              isComplete={isComplete || (!isStreamingMode && Boolean(answer))} 
            />
          )}
        </div>

        <div className="sticky bottom-12 z-10">
          <div className="glass-card p-2 p-3 bg-zinc-50/95 dark:bg-zinc-900/95 shadow-xl shadow-zinc-200/50 dark:shadow-none">
            
            {showFilters && (
              <div className="px-3 pb-3 pt-1 mb-3 border-b border-zinc-200 dark:border-zinc-800 flex flex-wrap gap-4 animate-in slide-in-from-top-2 duration-300">
                <label className="flex flex-col gap-1 text-xs font-semibold text-zinc-600 dark:text-zinc-400">
                  Document Type
                  <select 
                    value={docTypeFilter} 
                    onChange={e => setDocTypeFilter(e.target.value as any)}
                    className="mt-1 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-md py-1.5 px-3 focus:outline-none focus:border-primary text-sm font-medium"
                  >
                    <option value="all">All Types</option>
                    {DOC_TYPES.map(t => <option key={t} value={t} className="capitalize">{t}</option>)}
                  </select>
                </label>
                
                <label className="flex flex-col gap-1 text-xs font-semibold text-zinc-600 dark:text-zinc-400">
                  Session ID Limit
                  <input 
                    type="text" 
                    value={sessionFilter} 
                    onChange={e => setSessionFilter(e.target.value)}
                    placeholder="e.g. uuid-string"
                    className="mt-1 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-md py-1.5 px-3 focus:outline-none focus:border-primary text-sm font-medium w-64"
                  />
                </label>
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex items-end gap-2">
              <button 
                type="button" 
                onClick={() => setShowFilters(!showFilters)}
                className="p-3 text-zinc-400 hover:text-primary transition-colors flex-shrink-0 bg-transparent rounded-xl"
                title="Filters"
              >
                <Settings2 className="w-5 h-5" />
              </button>
              
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
                <div className="px-3 pb-2 flex items-center justify-between">
                  <label className="flex items-center gap-2 cursor-pointer group">
                    <input 
                      type="checkbox" 
                      className="peer sr-only" 
                      checked={isStreamingMode}
                      onChange={() => setIsStreamingMode(!isStreamingMode)}
                    />
                    <div className="w-7 h-4 bg-zinc-200 dark:bg-zinc-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-primary relative" />
                    <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider group-hover:text-zinc-800 dark:group-hover:text-zinc-200 transition-colors">Stream</span>
                  </label>
                </div>
              </div>
              
              <button 
                type="submit" 
                disabled={!query.trim() || isAnswering}
                className="bg-primary hover:bg-primary-dark text-white p-3.5 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all flex-shrink-0"
              >
                {isAnswering ? <div className="w-5 h-5 border-2 border-white/30 border-t-white animate-spin rounded-full" /> : <Send className="w-5 h-5 -ml-0.5 mt-0.5" />}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
