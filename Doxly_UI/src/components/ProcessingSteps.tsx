import React, { useEffect, useState } from "react";
import { clsx } from "clsx";
import { Check, Loader2 } from "lucide-react";

const steps = [
  "Classifying document type...",
  "Resolving coreferences...",
  "Chunking and embedding...",
  "Storing in knowledge graph..."
];

export function ProcessingSteps({ isComplete }: { isComplete: boolean }) {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (isComplete) {
      setCurrentStep(steps.length);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev < steps.length - 1 ? prev + 1 : prev));
    }, 3000); // Faux progress since it's a blocking single backend call

    return () => clearInterval(interval);
  }, [isComplete]);

  return (
    <div className="w-full max-w-sm mx-auto mt-8 space-y-4">
      {steps.map((step, idx) => {
        const isDone = idx < currentStep || isComplete;
        const isActive = idx === currentStep && !isComplete;
        const isPending = idx > currentStep && !isComplete;

        return (
          <div
            key={idx}
            className={clsx(
              "flex items-center gap-3 transition-opacity duration-500",
              isPending ? "opacity-40" : "opacity-100"
            )}
          >
            <div
              className={clsx(
                "w-6 h-6 rounded-full flex items-center justify-center border text-xs",
                isDone && "bg-emerald-500 border-emerald-500 text-white",
                isActive && "border-primary text-primary",
                isPending && "border-zinc-300 dark:border-zinc-700 text-transparent"
              )}
            >
              {isDone ? (
                <Check className="w-3.5 h-3.5" />
              ) : isActive ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <div className="w-1.5 h-1.5 rounded-full bg-zinc-300 dark:bg-zinc-700" />
              )}
            </div>
            <span
              className={clsx(
                "text-sm font-medium",
                isDone && "text-emerald-700 dark:text-emerald-400",
                isActive && "text-zinc-900 dark:text-zinc-100",
                isPending && "text-zinc-400 dark:text-zinc-600"
              )}
            >
              {step}
            </span>
          </div>
        );
      })}
    </div>
  );
}
