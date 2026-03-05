/** API client for communicating with the backend. */

const API_BASE = "http://localhost:8000";

export async function createRun(file: File): Promise<{ run_id: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/runs`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Create run failed: ${res.statusText}`);
  return res.json();
}

export async function listRuns(): Promise<{ runs: Array<{ run_id: string; status: string; stage: string; created_at?: string }> }> {
  const res = await fetch(`${API_BASE}/api/runs`);
  if (!res.ok) throw new Error(`List runs failed: ${res.statusText}`);
  return res.json();
}

export async function stopRun(runId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`Stop run failed: ${res.statusText}`);
  return res.json();
}

export async function resumeRun(
  runId: string,
  decision: Record<string, unknown>
): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/resume`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision }),
  });
  if (!res.ok) throw new Error(`Resume run failed: ${res.statusText}`);
  return res.json();
}

export async function getReport(runId: string): Promise<{ run_id: string; report?: string; error?: string }> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/report`);
  if (!res.ok) throw new Error(`Get report failed: ${res.statusText}`);
  return res.json();
}

export function getStreamUrl(runId: string): string {
  return `${API_BASE}/api/runs/${runId}/stream`;
}
