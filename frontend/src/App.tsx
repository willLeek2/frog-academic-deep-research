import { useEffect, useState, useCallback } from "react";
import RunList from "./components/RunList";
import RunCreator from "./components/RunCreator";
import ProgressPanel from "./components/ProgressPanel";
import { listRuns } from "./api/client";
import type { RunSummary } from "./types";

function App() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      const resp = await listRuns();
      setRuns(resp.runs);
    } catch {
      // backend may not be running yet
    }
  }, []);

  useEffect(() => {
    refreshRuns();
    const interval = setInterval(refreshRuns, 5000);
    return () => clearInterval(interval);
  }, [refreshRuns]);

  const handleCreated = (runId: string) => {
    setSelectedRunId(runId);
    refreshRuns();
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-72 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col">
        <div className="px-4 py-4 border-b border-gray-200">
          <h1 className="text-lg font-bold text-gray-800">
            🐸 Deep Research
          </h1>
          <p className="text-xs text-gray-400 mt-0.5">Academic Research Agent</p>
        </div>

        <RunCreator onCreated={handleCreated} />

        <div className="flex-1 overflow-auto">
          <div className="px-4 py-2">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              History
            </h2>
          </div>
          <RunList
            runs={runs}
            selectedRunId={selectedRunId}
            onSelect={setSelectedRunId}
          />
        </div>
      </aside>

      {/* Main panel */}
      <main className="flex-1 overflow-hidden">
        <ProgressPanel runId={selectedRunId} />
      </main>
    </div>
  );
}

export default App;
