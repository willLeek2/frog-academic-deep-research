import { useEffect, useRef, useState, useCallback } from "react";
import type { RunState } from "../types";
import { getStreamUrl } from "../api/client";

/**
 * Custom hook that connects to the backend SSE endpoint and streams
 * real-time progress updates for a given run.
 */
export function useSSE(runId: string | null) {
  const [state, setState] = useState<RunState | null>(null);
  const [connected, setConnected] = useState(false);
  const sourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setConnected(false);
  }, []);

  useEffect(() => {
    if (!runId) {
      disconnect();
      return;
    }

    const url = getStreamUrl(runId);
    const source = new EventSource(url);
    sourceRef.current = source;

    source.addEventListener("progress", (event) => {
      try {
        const data: RunState = JSON.parse(event.data);
        setState(data);
      } catch {
        // ignore malformed events
      }
    });

    source.addEventListener("done", () => {
      disconnect();
    });

    source.onopen = () => setConnected(true);

    source.onerror = () => {
      // EventSource will auto-reconnect; we just update state
      setConnected(false);
    };

    return () => {
      disconnect();
    };
  }, [runId, disconnect]);

  return { state, connected, disconnect };
}
