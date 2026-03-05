"""Stage 5A: Outline planner agent.

Takes all research notes and produces a structured outline JSON for the
final report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import ResearchState
from utils.llm_helpers import extract_text, parse_json_response
from utils.run_logger import RunLogger

_SYSTEM = (
    "You are a research report architect. Given research notes and path "
    "evaluations, create a detailed report outline. Output ONLY a JSON object:\n"
    '{\n'
    '  "title": "<report title>",\n'
    '  "sections": [\n'
    '    {\n'
    '      "id": "sec-<N>",\n'
    '      "title": "<section title>",\n'
    '      "level": 1,\n'
    '      "target_words": <approximate word count>,\n'
    '      "related_paths": ["path-1", ...],\n'
    '      "description": "<what this section covers>"\n'
    '    }, ...\n'
    '  ]\n'
    "}\n"
    "Include an Introduction, main body sections based on the research paths, "
    "a section for suboptimal/alternative approaches, and a Conclusion. "
    "Allocate more words to valuable paths."
)


class OutlinePlannerAgent:
    """Callable LangGraph node: outline planning."""

    def __init__(
        self,
        llm: BaseChatModel,
        run_logger: RunLogger,
        run_dir: str | Path,
    ) -> None:
        self._llm = llm
        self._logger = run_logger
        self._run_dir = Path(run_dir)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        notes = state.get("research_notes", {})
        evaluations = state.get("path_evaluations", [])
        ctx = state.get("extracted_context") or {}
        topic = ctx.get("topic", "unknown")
        paths = state.get("research_paths", [])

        outline = self._generate_outline(topic, paths, evaluations, notes)

        # Persist
        writing_dir = self._run_dir / "writing"
        writing_dir.mkdir(parents=True, exist_ok=True)
        (writing_dir / "outline.json").write_text(
            json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._logger.log("writing", "outline_planner", "outline_generated",
                         {"sections": len(outline.get("sections", []))})

        return {"outline": outline, "current_stage": "outline_planning_done"}

    # ------------------------------------------------------------------
    def _generate_outline(
        self,
        topic: str,
        paths: list[dict],
        evaluations: list[dict],
        notes: dict[str, str],
    ) -> dict:
        # Build a compact summary of all notes
        notes_summary = ""
        for pid, note in notes.items():
            notes_summary += f"### Path {pid}\n{note[:1000]}\n\n"

        eval_summary = json.dumps(
            [{"path_id": e.get("path_id"), "score": e.get("score"),
              "category": e.get("category")} for e in evaluations],
            ensure_ascii=False,
        )

        prompt = (
            f"Research topic: {topic}\n\n"
            f"Path evaluations:\n{eval_summary}\n\n"
            f"Research notes summary:\n{notes_summary[:6000]}\n\n"
            "Create a detailed report outline."
        )
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, dict) and "sections" in parsed:
                return parsed
        except Exception:
            pass
        # Fallback
        return {
            "title": f"Research Report: {topic}",
            "sections": [
                {"id": "sec-1", "title": "Introduction", "level": 1,
                 "target_words": 500, "related_paths": [], "description": "Overview"},
                {"id": "sec-2", "title": "Background", "level": 1,
                 "target_words": 800, "related_paths": [], "description": "Background"},
                {"id": "sec-3", "title": "Main Findings", "level": 1,
                 "target_words": 1500, "related_paths": [p.get("id") for p in paths],
                 "description": "Core research findings"},
                {"id": "sec-4", "title": "Alternative Approaches", "level": 1,
                 "target_words": 500, "related_paths": [], "description": "Suboptimal paths"},
                {"id": "sec-5", "title": "Conclusion", "level": 1,
                 "target_words": 400, "related_paths": [], "description": "Summary"},
            ],
        }
