"""Stage 3: Path evaluator agent.

Evaluates all research paths from the broad survey, scores them, and
classifies each as *valuable* or *suboptimal*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import ResearchState
from utils.llm_helpers import extract_text, parse_json_response
from utils.mcp_caller import MCPCaller
from utils.run_logger import RunLogger

_SYSTEM = (
    "You are a research path evaluator. Given a list of research paths and "
    "their descriptions, evaluate each path. Output ONLY a JSON array of "
    "objects, each with:\n"
    '  "path_id": "<id>",\n'
    '  "score": <0.0-1.0>,\n'
    '  "category": "valuable"|"suboptimal",\n'
    '  "reason": "<brief reason>",\n'
    '  "suggested_depth": "deep"|"brief"\n'
    "\nPaths with score >= 0.6 should be 'valuable', otherwise 'suboptimal'."
)


class PathEvaluatorAgent:
    """Callable LangGraph node: path evaluation and classification."""

    def __init__(
        self,
        llm: BaseChatModel,
        mcp_caller: MCPCaller,
        run_logger: RunLogger,
        run_dir: str | Path,
    ) -> None:
        self._llm = llm
        self._mcp = mcp_caller
        self._logger = run_logger
        self._run_dir = Path(run_dir)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        paths = state.get("research_paths", [])
        ctx = state.get("extracted_context") or {}
        topic = ctx.get("topic", "unknown")

        evaluations = self._evaluate(topic, paths)

        # Persist assessments
        for ev in evaluations:
            pid = ev.get("path_id", "unknown")
            pdir = self._run_dir / "paths" / pid
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "assessment.json").write_text(
                json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        self._logger.log("path_evaluation", "path_evaluator", "evaluated",
                         {"count": len(evaluations),
                          "valuable": sum(1 for e in evaluations if e.get("category") == "valuable"),
                          "suboptimal": sum(1 for e in evaluations if e.get("category") == "suboptimal")})

        return {"path_evaluations": evaluations, "current_stage": "path_evaluation_done"}

    # ------------------------------------------------------------------
    def _evaluate(self, topic: str, paths: list[dict]) -> list[dict]:
        paths_desc = json.dumps(
            [{"id": p.get("id"), "title": p.get("title"), "description": p.get("description")}
             for p in paths],
            ensure_ascii=False,
        )
        prompt = (
            f"Research topic: {topic}\n\n"
            f"Research paths to evaluate:\n{paths_desc}"
        )
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, list):
                # Normalise
                for ev in parsed:
                    score = float(ev.get("score", 0.5))
                    ev["score"] = max(0.0, min(1.0, score))
                    if "category" not in ev:
                        ev["category"] = "valuable" if score >= 0.6 else "suboptimal"
                return parsed
        except Exception:
            pass
        # Fallback: all valuable
        return [
            {"path_id": p.get("id", f"path-{i}"), "score": 0.85,
             "category": "valuable", "reason": "Default evaluation.",
             "suggested_depth": "deep"}
            for i, p in enumerate(paths)
        ]
