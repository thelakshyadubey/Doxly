import React, { useRef, useState } from "react";
import { UploadCloud, File, LogOut, ArrowRight, XCircle, Check } from "lucide-react";
import { SessionBadge } from "./SessionBadge";
import { ProcessingSteps } from "./ProcessingSteps";
import { useToast } from "../hooks/useToast";

export function UploadScreen({
  userId,
  session,
  onLogout,
  onProceed,
  onSkipToChat,
}: {
  userId: string;
  session: any; // ReturnType<typeof useSession>
  onLogout: () => void;
  onProceed: () => void;
  onSkipToChat: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const { addToast } = useToast();
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; size: number }[]>([]);

  const handleFileDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    if (session.status === 'processing' || session.status === 'indexed') return;
    if (e.dataTransfer.files?.length) {
      await processFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (session.status === 'processing' || session.status === 'indexed') return;
    if (e.target.files?.length) {
      await processFiles(e.target.files);
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const processFiles = async (files: FileList) => {
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.type.match(/(image\/jpeg|image\/png|image\/tiff|application\/pdf)/)) {
        addToast(`Invalid file type: ${file.name}. Only JPEG, PNG, TIFF, or PDF pages allowed.`, "error");
        continue;
      }
      
      try {
        await session.uploadFile(file);
        setUploadedFiles(prev => [...prev, { name: file.name, size: file.size }]);
        addToast(`Successfully uploaded ${file.name}`, "success");
      } catch (err) {
        // useSession already shows toast on err, we just skip it
      }
    }
  };

  const handleProcess = async () => {
    try {
      await session.flushSession();
      addToast("Documents processed successfully!", "success");
    } catch {
      // toast shown in hook
    }
  };

  const handleStartNew = () => {
    session.resetSession();
    setUploadedFiles([]);
  };

  return (
    <div className="flex flex-col min-h-screen px-4 py-8 animate-in fade-in duration-300 max-w-4xl mx-auto w-full pb-24">
      <header className="flex items-center justify-between mb-8 pb-4 border-b border-zinc-200 dark:border-zinc-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Doxly Upload</h1>
          <p className="text-sm text-zinc-500">{userId}</p>
        </div>
        <div className="flex items-center gap-4">
          <SessionBadge status={session.status} />
          <button
            onClick={onLogout}
            className="flex items-center gap-2 text-sm font-medium text-zinc-500 hover:text-red-500 transition-colors"
          >
            <LogOut className="w-4 h-4" /> Revoke
          </button>
        </div>
      </header>

      {session.status === "indexed" ? (
        <div className="glass-card p-8 text-center animate-in zoom-in-95 duration-500 border-emerald-500/20">
          <div className="w-16 h-16 bg-emerald-100 dark:bg-emerald-500/20 text-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-inner border border-emerald-500/20">
             <Check className="w-8 h-8" /> 
             {/* Missing Check icon, fallback applied above mentally, wait I need to import it */}
          </div>
          <h2 className="text-2xl font-semibold mb-2 text-emerald-950 dark:text-emerald-50">Analysis Complete</h2>
          <div className="flex justify-center gap-6 text-sm mb-8 text-emerald-800 dark:text-emerald-200">
            <div>
              <span className="block text-2xl font-bold text-emerald-600 dark:text-emerald-400">{session.docType}</span>
              <span className="opacity-70 uppercase tracking-wider text-[10px] font-bold">Doc Type</span>
            </div>
            <div>
              <span className="block text-2xl font-bold text-emerald-600 dark:text-emerald-400">{session.chunkCount}</span>
              <span className="opacity-70 uppercase tracking-wider text-[10px] font-bold">Chunks</span>
            </div>
            <div>
              <span className="block text-2xl font-bold text-emerald-600 dark:text-emerald-400">{session.entityCount}</span>
              <span className="opacity-70 uppercase tracking-wider text-[10px] font-bold">Entities</span>
            </div>
          </div>
          
          <div className="flex justify-center gap-4">
            <button
              onClick={handleStartNew}
              className="px-6 py-2.5 rounded-lg border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-800 font-medium transition-colors"
            >
              Start New Batch
            </button>
            <button
              onClick={onProceed}
              className="px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-white font-medium transition-colors shadow-sm flex items-center gap-2"
            >
              Ask Questions <ArrowRight className="w-4 h-4"/>
            </button>
          </div>
        </div>
      ) : session.status === "processing" ? (
        <div className="glass-card p-12 w-full flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-zinc-200 dark:border-zinc-800 border-t-primary rounded-full animate-spin mb-8" />
          <h2 className="text-xl font-medium mb-2">Analyzing Documents</h2>
          <p className="text-zinc-500 text-sm max-w-sm text-center mb-4">
            Our AI is currently running OCR, chunking, and knowledge graph mapping. This may take up to 30 seconds.
          </p>
          <ProcessingSteps isComplete={false} />
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              onClick={() => fileRef.current?.click()}
              className="border-2 border-dashed border-zinc-300 dark:border-zinc-700 hover:border-primary dark:hover:border-primary hover:bg-primary/5 transition-all rounded-2xl flex flex-col items-center justify-center p-12 cursor-pointer bg-white/50 dark:bg-zinc-900/50 min-h-[300px]"
            >
              <input
                type="file"
                multiple
                ref={fileRef}
                className="hidden"
                onChange={handleFileSelect}
                accept="image/jpeg,image/png,image/tiff,application/pdf"
              />
              <UploadCloud className="w-12 h-12 text-zinc-400 mb-4" />
              <p className="text-zinc-700 dark:text-zinc-300 font-medium text-lg">Click or drag documents here</p>
              <p className="text-zinc-500 text-sm mt-2">JPEG, PNG, TIFF, or single-page PDF</p>
              <button
                onClick={(e) => { e.stopPropagation(); onSkipToChat(); }}
                className="mt-6 text-xs text-primary hover:underline font-medium"
              >
                Skip — just chat without documents
              </button>
              {session.isUploading && (
                <div className="mt-6 flex items-center gap-2 text-primary font-medium text-sm">
                  <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                  Uploading...
                </div>
              )}
            </div>
            {session.status === "failed" && (
               <div className="mt-4 p-4 rounded-xl bg-red-50 text-red-800 flex items-start gap-3 border border-red-200 text-sm font-medium">
                 <XCircle className="w-5 h-5 flex-shrink-0" />
                 <div>
                   Processing failed. Please try again or start a new session.
                   <div className="mt-2 text-red-600 cursor-pointer underline underline-offset-2" onClick={handleProcess}>
                     Retry Processing
                   </div>
                 </div>
               </div>
            )}
          </div>
          
          <div className="flex flex-col h-full">
            <div className="glass-card flex-1 flex flex-col p-4">
              <div className="flex items-center justify-between mb-4 border-b border-zinc-100 dark:border-zinc-800 pb-2">
                <h3 className="font-semibold text-sm">Uploaded Pages</h3>
                <span className="bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 rounded-full text-xs font-mono text-zinc-600 dark:text-zinc-400">
                  {session.pageCount} total
                </span>
              </div>
              
              <div className="flex-1 overflow-y-auto space-y-2 mb-4 max-h-[300px]">
                {uploadedFiles.length === 0 && (
                  <p className="text-xs text-zinc-400 text-center py-8">No files uploaded yet.</p>
                )}
                {uploadedFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-3 p-2 rounded-lg bg-zinc-50 dark:bg-zinc-800/50 border border-zinc-100 dark:border-zinc-800">
                    <File className="w-4 h-4 text-zinc-400" />
                    <div className="flex-1 truncate">
                      <p className="text-xs font-medium truncate">{f.name}</p>
                      <p className="text-[10px] text-zinc-400">{(f.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="pt-4 border-t border-zinc-100 dark:border-zinc-800 mt-auto flex flex-col gap-2">
                <button
                  onClick={handleProcess}
                  disabled={session.pageCount === 0 || session.isUploading}
                  className="w-full bg-primary hover:bg-primary-dark disabled:bg-zinc-300 disabled:text-zinc-500 text-white font-medium py-2.5 rounded-lg transition-colors flex items-center justify-center gap-2"
                >
                  Process Documents
                </button>
                {session.pageCount > 0 && (
                   <button
                    onClick={handleStartNew}
                    className="text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-300 pt-2 font-medium"
                   >
                     Clear & restart
                   </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
