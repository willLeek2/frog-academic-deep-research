import { STAGES, STAGE_LABELS } from "../types";

interface StageIndicatorProps {
  currentStage: string;
}

/**
 * Visual indicator showing the six pipeline stages.
 * Highlights completed, active, and pending stages.
 */
export default function StageIndicator({ currentStage }: StageIndicatorProps) {
  const getStageStatus = (stageName: string): "completed" | "active" | "pending" => {
    const currentIdx = STAGES.indexOf(currentStage as (typeof STAGES)[number]);
    const stageIdx = STAGES.indexOf(stageName as (typeof STAGES)[number]);

    // Handle terminal states
    if (currentStage === "completed" || currentStage === "stopped" || currentStage === "error") {
      return "completed";
    }

    // Handle "_done" suffixed stages
    const doneMatch = currentStage.match(/^(.+)_done$/);
    if (doneMatch) {
      const baseStage = doneMatch[1];
      const baseIdx = STAGES.indexOf(baseStage as (typeof STAGES)[number]);
      if (stageIdx < baseIdx) return "completed";
      if (stageIdx === baseIdx) return "completed";
      return "pending";
    }

    if (currentIdx === -1) {
      // Current stage may be a sub-stage; try matching by prefix
      for (let i = STAGES.length - 1; i >= 0; i--) {
        if (currentStage.startsWith(STAGES[i])) {
          if (stageIdx < i) return "completed";
          if (stageIdx === i) return "active";
          return "pending";
        }
      }
      return "pending";
    }

    if (stageIdx < currentIdx) return "completed";
    if (stageIdx === currentIdx) return "active";
    return "pending";
  };

  return (
    <div className="flex items-center gap-2 py-3 px-4 overflow-x-auto">
      {STAGES.map((stage, idx) => {
        const status = getStageStatus(stage);
        return (
          <div key={stage} className="flex items-center">
            {idx > 0 && (
              <div
                className={`w-6 h-0.5 mx-1 ${
                  status === "pending" ? "bg-gray-300" : "bg-blue-500"
                }`}
              />
            )}
            <div className="flex flex-col items-center min-w-[80px]">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold
                  ${status === "completed" ? "bg-green-500 text-white" : ""}
                  ${status === "active" ? "bg-blue-500 text-white animate-pulse" : ""}
                  ${status === "pending" ? "bg-gray-200 text-gray-500" : ""}
                `}
              >
                {status === "completed" ? "✓" : idx + 1}
              </div>
              <span
                className={`text-[10px] mt-1 text-center leading-tight
                  ${status === "active" ? "font-semibold text-blue-600" : "text-gray-500"}
                `}
              >
                {STAGE_LABELS[stage] || stage}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
