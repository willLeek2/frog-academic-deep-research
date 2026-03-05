import { useSSE } from "../hooks/useSSE";
import { stopRun, getReport } from "../api/client";
import StageIndicator from "./StageIndicator";
import { useState } from "react";

interface ProgressPanelProps {
  runId: string | null;
}

export default function ProgressPanel({ runId }: ProgressPanelProps) {
  const { state, connected } = useSSE(runId);
  const [report, setReport] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  if (!runId) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">
        <p>Select or create a research task to view progress.</p>
      </div>
    );
  }

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopRun(runId);
    } finally {
      setStopping(false);
    }
  };

  const handleViewReport = async () => {
    try {
      const resp = await getReport(runId);
      setReport(resp.report || resp.error || "No report available.");
    } catch {
      setReport("Failed to load report.");
    }
  };

  const isRunning = state?.status === "running";
  const isCompleted = state?.status === "completed";
  const isTerminal = ["completed", "error", "stopped"].includes(state?.status || "");

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-800">Run: {runId}</h2>
          <div className="flex items-center gap-2 mt-1">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                connected ? "bg-green-400" : "bg-gray-300"
              }`}
            />
            <span className="text-xs text-gray-500">
              {connected ? "Connected" : isTerminal ? "Finished" : "Disconnected"}
            </span>
            {state && (
              <span className="text-xs text-gray-500 ml-2">
                Status: <strong>{state.status}</strong>
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          {isRunning && (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="px-4 py-2 text-sm font-medium rounded bg-red-500 text-white hover:bg-red-600 disabled:bg-red-300 transition-colors"
            >
              {stopping ? "Stopping..." : "Emergency Stop"}
            </button>
          )}
          {isCompleted && (
            <button
              onClick={handleViewReport}
              className="px-4 py-2 text-sm font-medium rounded bg-green-500 text-white hover:bg-green-600 transition-colors"
            >
              View Report
            </button>
          )}
        </div>
      </div>

      {/* Stage Indicator */}
      {state && (
        <div className="border-b border-gray-100">
          <StageIndicator currentStage={state.stage} />
        </div>
      )}

      {/* Details */}
      <div className="flex-1 overflow-auto p-6">
        {state ? (
          <div className="space-y-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-600 mb-2">Current State</h3>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                <dt className="text-gray-500">Stage</dt>
                <dd className="font-mono">{state.stage}</dd>
                <dt className="text-gray-500">Status</dt>
                <dd className="font-mono">{state.status}</dd>
                {state.created_at && (
                  <>
                    <dt className="text-gray-500">Created</dt>
                    <dd className="font-mono text-xs">{state.created_at}</dd>
                  </>
                )}
                {state.error && (
                  <>
                    <dt className="text-gray-500">Error</dt>
                    <dd className="text-red-500 text-xs">{state.error}</dd>
                  </>
                )}
              </dl>
            </div>

            {report && (
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-gray-600 mb-2">Report</h3>
                <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono max-h-96 overflow-auto">
                  {report}
                </pre>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center h-32 text-gray-400">
            <p>Connecting to run...</p>
          </div>
        )}
      </div>
    </div>
  );
}
