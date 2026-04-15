import React, { createContext, useContext, useState, useCallback } from "react";
import type { ReactNode } from "react";
import { X, CheckCircle, AlertCircle, Info } from "lucide-react";
import { clsx } from "clsx";

export type ToastType = "success" | "error" | "info";

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextType {
  addToast: (message: string, type: ToastType) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: ToastType) => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={clsx(
              "flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg border text-sm font-medium animate-in slide-in-from-bottom-5 fade-in duration-300",
              toast.type === "success" && "bg-emerald-50 text-emerald-900 border-emerald-200 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20",
              toast.type === "error" && "bg-red-50 text-red-900 border-red-200 dark:bg-red-500/10 dark:text-red-400 dark:border-red-500/20",
              toast.type === "info" && "bg-blue-50 text-blue-900 border-blue-200 dark:bg-blue-500/10 dark:text-blue-400 dark:border-blue-500/20"
            )}
          >
            {toast.type === "success" && <CheckCircle className="w-5 h-5 flex-shrink-0" />}
            {toast.type === "error" && <AlertCircle className="w-5 h-5 flex-shrink-0" />}
            {toast.type === "info" && <Info className="w-5 h-5 flex-shrink-0" />}
            <span className="flex-1">{toast.message}</span>
            <button
              onClick={() => removeToast(toast.id)}
              className="text-inherit opacity-70 hover:opacity-100 transition-opacity"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used within ToastProvider");
  return context;
}
