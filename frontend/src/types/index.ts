/** TypeScript type definitions for the deep research agent frontend. */

export interface RunState {
  run_id: string;
  status: "running" | "completed" | "error" | "stopped" | "paused";
  stage: string;
  created_at?: string;
  progress?: Record<string, unknown>;
  error?: string | null;
}

export interface RunSummary {
  run_id: string;
  status: string;
  stage: string;
  created_at?: string;
}

export interface CreateRunResponse {
  run_id: string;
  status: string;
}

export interface ResumeRequest {
  decision: Record<string, unknown>;
}

export interface ReportResponse {
  run_id: string;
  report?: string;
  error?: string;
}

export interface ListRunsResponse {
  runs: RunSummary[];
}

export const STAGES = [
  "input_preprocessing",
  "broad_survey",
  "path_evaluation",
  "deep_research",
  "writing",
  "post_processing",
] as const;

export type StageName = (typeof STAGES)[number];

export const STAGE_LABELS: Record<string, string> = {
  input_preprocessing: "Input Preprocessing",
  broad_survey: "Broad Survey",
  path_evaluation: "Path Evaluation",
  deep_research: "Deep Research",
  writing: "Writing",
  post_processing: "Post Processing",
};
