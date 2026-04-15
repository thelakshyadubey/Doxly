import React, { useState } from "react";
import { ArrowRight, FileText } from "lucide-react";
import { api } from "../api/api-client";
import { useToast } from "../hooks/useToast";

export function IdentityScreen({
  onAuthComplete,
  setUserId,
}: {
  onAuthComplete: () => void;
  setUserId: (id: string) => void;
}) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const { addToast } = useToast();

  const handleContinue = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !email.includes("@")) {
      addToast("Please enter a valid email", "error");
      return;
    }

    setLoading(true);
    setUserId(email);
    try {
      const { authorized } = await api.auth.status(email);
      if (authorized) {
        onAuthComplete();
      } else {
        // Redraw will show ConnectScreen because we setUserId but it's not authorized
        // Wait, App.tsx will handle the routing once userId is set but not authorized
      }
    } catch (err) {
      addToast("Failed to verify user status", "error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] px-4 animate-in fade-in zoom-in-95 duration-500">
      <div className="glass-card w-full max-w-md p-8 text-center relative overflow-hidden group">
        <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />
        
        <div className="w-16 h-16 bg-primary/10 dark:bg-primary/20 text-primary rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-inner border border-primary/20">
          <FileText className="w-8 h-8" />
        </div>
        
        <h1 className="text-2xl font-semibold tracking-tight mb-2">Welcome to Doxly</h1>
        <p className="text-zinc-500 dark:text-zinc-400 mb-8 text-sm">
          Intelligent document analysis powered by AI. Please enter your email to get started.
        </p>

        <form onSubmit={handleContinue} className="space-y-4">
          <div className="text-left">
            <label htmlFor="email" className="sr-only">Email address</label>
            <input
              id="email"
              type="email"
              placeholder="name@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 rounded-xl bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all placeholder:text-zinc-400 dark:placeholder:text-zinc-600"
              required
            />
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-zinc-900 hover:bg-zinc-800 text-white dark:bg-zinc-100 dark:hover:bg-white dark:text-zinc-950 font-medium py-3 rounded-xl transition-all flex items-center justify-center gap-2 group disabled:opacity-70 disabled:cursor-not-allowed"
          >
            {loading ? "Checking..." : "Continue"}
            {!loading && <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />}
          </button>
        </form>
      </div>
    </div>
  );
}
