import type { RunSummary } from "../types";

interface RunListProps {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
}

const STATUS_COLORS: Record<string, string> = {
  running: "bg-blue-100 text-blue-700",
  completed: "bg-green-100 text-green-700",
  error: "bg-red-100 text-red-700",
  stopped: "bg-yellow-100 text-yellow-700",
  paused: "bg-purple-100 text-purple-700",
};

export default function RunList({ runs, selectedRunId, onSelect }: RunListProps) {
  if (runs.length === 0) {
    return (
      <div className="p-4 text-gray-400 text-sm text-center">
        No runs yet. Create a new research task to get started.
      </div>
    );
  }

  return (
    <ul className="divide-y divide-gray-100">
      {runs.map((run) => (
        <li
          key={run.run_id}
          onClick={() => onSelect(run.run_id)}
          className={`px-4 py-3 cursor-pointer hover:bg-gray-50 transition-colors ${
            selectedRunId === run.run_id ? "bg-blue-50 border-l-4 border-blue-500" : ""
          }`}
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-sm text-gray-700">{run.run_id}</span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full ${
                STATUS_COLORS[run.status] || "bg-gray-100 text-gray-600"
              }`}
            >
              {run.status}
            </span>
          </div>
          <div className="text-xs text-gray-400 mt-1">{run.stage}</div>
        </li>
      ))}
    </ul>
  );
}
