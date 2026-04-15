import { useState, useEffect } from "react";

export function useUser() {
  const [userId, setUserIdState] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("doxly_user_id");
    if (stored) {
      setUserIdState(stored);
    }
    setIsLoaded(true);
  }, []);

  const setUserId = (id: string | null) => {
    if (id) {
      localStorage.setItem("doxly_user_id", id);
    } else {
      localStorage.removeItem("doxly_user_id");
    }
    setUserIdState(id);
  };

  const logout = () => {
    setUserId(null);
  };

  return { userId, setUserId, logout, isLoaded };
}
