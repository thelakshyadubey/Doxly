import React, { useState, useEffect } from "react";
import { StatusBar } from "./components/StatusBar";
import { ServerOfflineBanner } from "./components/ServerOfflineBanner";
import { ConnectScreen } from "./components/ConnectScreen";
import { UploadScreen } from "./components/UploadScreen";
import { QueryScreen } from "./components/QueryScreen";
import { useHealth } from "./hooks/useHealth";
import { useUser } from "./hooks/useUser";
import { useSession } from "./hooks/useSession";
import { ToastProvider, useToast } from "./hooks/useToast";
import { api } from "./api/api-client";

function AppContent() {
  const { isLive } = useHealth();
  const { userId, setUserId, logout, isLoaded } = useUser();
  const session = useSession(userId);
  const { addToast } = useToast();

  const [isAuthorized, setIsAuthorized] = useState<boolean | null>(null);
  const [currentScreen, setCurrentScreen] = useState<"connect" | "upload" | "query">("connect");

  // After Google OAuth redirect, the backend appends ?user_id=email to the URL.
  // Capture it immediately so the rest of the app can use it.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const oauthUserId = params.get("user_id");
    if (oauthUserId) {
      setUserId(oauthUserId);
      // Clean the URL — no reload needed
      window.history.replaceState({}, "", window.location.pathname);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Whenever userId changes (on load from localStorage or after OAuth redirect),
  // verify authorization status with the backend.
  useEffect(() => {
    if (!isLoaded) return;
    if (userId) {
      checkAuthStatus(userId);
    } else {
      setCurrentScreen("connect");
      setIsAuthorized(null);
    }
  }, [userId, isLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  const checkAuthStatus = async (id: string) => {
    try {
      const res = await api.auth.status(id);
      setIsAuthorized(res.authorized);
      setCurrentScreen(res.authorized ? "upload" : "connect");
    } catch (err) {
      console.error(err);
      addToast("Failed to check auth status", "error");
      setIsAuthorized(false);
      setCurrentScreen("connect");
    }
  };

  const handleRevoke = async () => {
    if (!userId) return;
    try {
      await api.auth.revoke(userId);
      logout();
      session.resetSession();
      setIsAuthorized(false);
      setCurrentScreen("connect");
      addToast("Access revoked successfully", "success");
    } catch {
      addToast("Failed to revoke access", "error");
    }
  };

  if (isLive === false) {
    return <ServerOfflineBanner />;
  }

  // Show spinner while localStorage is loading or while we're waiting for auth check
  if (!isLoaded || (userId && isAuthorized === null)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 rounded-full border-2 border-primary border-t-transparent animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen selection:bg-primary/20 selection:text-primary-dark">
      {currentScreen === "connect" && (
        <ConnectScreen />
      )}

      {currentScreen === "upload" && userId && (
        <UploadScreen
          userId={userId}
          session={session}
          onLogout={handleRevoke}
          onProceed={() => setCurrentScreen("query")}
          onSkipToChat={() => setCurrentScreen("query")}
        />
      )}

      {currentScreen === "query" && userId && (
        <QueryScreen
          userId={userId}
          currentSessionId={session.sessionId}
          onLogout={handleRevoke}
          onUploadMore={() => { session.resetSession(); setCurrentScreen("upload"); }}
        />
      )}

      <StatusBar />
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  );
}
