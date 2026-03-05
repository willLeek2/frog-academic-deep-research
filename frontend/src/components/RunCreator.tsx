import { useState, useRef } from "react";
import { createRun } from "../api/client";

interface RunCreatorProps {
  onCreated: (runId: string) => void;
}

export default function RunCreator({ onCreated }: RunCreatorProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFileChange = () => {
    const file = fileRef.current?.files?.[0];
    setFileName(file ? file.name : null);
    setError(null);
  };

  const handleSubmit = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setError("Please select a markdown file first.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const resp = await createRun(file);
      onCreated(resp.run_id);
      // Reset file input
      if (fileRef.current) fileRef.current.value = "";
      setFileName(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create run");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 border-b border-gray-200">
      <h3 className="text-sm font-semibold text-gray-600 mb-3">New Research Task</h3>

      <div className="space-y-3">
        <div>
          <label className="block">
            <span className="sr-only">Choose markdown file</span>
            <input
              ref={fileRef}
              type="file"
              accept=".md,.markdown,.txt"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-2 file:py-1.5 file:px-3
                file:rounded file:border-0
                file:text-sm file:font-medium
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100
                cursor-pointer"
            />
          </label>
          {fileName && (
            <p className="text-xs text-gray-500 mt-1 truncate">Selected: {fileName}</p>
          )}
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !fileName}
          className={`w-full py-2 px-4 rounded text-sm font-medium text-white transition-colors ${
            loading || !fileName
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Creating..." : "Create Research Task"}
        </button>

        {error && <p className="text-xs text-red-500">{error}</p>}
      </div>
    </div>
  );
}
