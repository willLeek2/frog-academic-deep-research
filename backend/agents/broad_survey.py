"""Stage 2: Broad survey agent.

For each research question, generates search queries via the heavy LLM, calls
MCP tools, and identifies distinct research paths.
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
from utils.stop_controller import StopController

_SEARCH_SYSTEM = (
    "You are a research search strategist. Given a research question, output "
    "ONLY a JSON object:\n"
    '{"queries": [{"query": "<search query>", "type": "perplexity"|"web"}], '
    '"max_queries": 3}'
)

_PATH_SYSTEM = (
    "You are a research path identifier. Given search results about a topic, "
    "identify distinct research paths/directions. Output ONLY a JSON array of "
    "objects, each with:\n"
    '  "id": "path-<N>",\n'
    '  "title": "<short descriptive title>",\n'
    '  "description": "<2-3 sentence description>",\n'
    '  "entry_papers": ["<url or title>", ...],\n'
    '  "status": "proposed"\n'
)


class BroadSurveyAgent:
    """Callable LangGraph node: broad survey with real MCP calls."""

    def __init__(
        self,
        llm: BaseChatModel,
        mcp_caller: MCPCaller,
        run_logger: RunLogger,
        stop_controller: StopController,
        run_dir: str | Path,
    ) -> None:
        self._llm = llm
        self._mcp = mcp_caller
        self._logger = run_logger
        self._stop = stop_controller
        self._run_dir = Path(run_dir)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        ctx = state.get("extracted_context") or {}
        topic = ctx.get("topic", "unknown")
        questions = ctx.get("questions", [topic])
        if not questions:
            questions = [topic]

        all_search_results: list[str] = []
        stage = "broad_survey"

        # Step 1: For each question, generate search queries & execute
        for q in questions:
            if self._stop.is_stop_requested():
                break

            queries = self._generate_queries(q)
            for qobj in queries:
                if self._stop.is_stop_requested():
                    break
                qtext = qobj.get("query", q)
                qtype = qobj.get("type", "perplexity")
                if qtype == "perplexity":
                    result = self._mcp.perplexity_search(qtext, stage)
                else:
                    result = self._mcp.web_search(qtext, stage)
                all_search_results.append(result)
                self._logger.log(stage, "broad_survey", "search_result",
                                 {"query": qtext, "type": qtype, "result_len": len(result)})

        # Step 2: Identify research paths from aggregated results
        paths = self._identify_paths(topic, questions, all_search_results)

        # Step 3: Save each path
        paths_dir = self._run_dir / "paths"
        paths_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            pdir = paths_dir / p["id"]
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "info.json").write_text(
                json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        self._logger.log(stage, "broad_survey", "paths_identified",
                         {"count": len(paths)})

        return {"research_paths": paths, "current_stage": "broad_survey_done"}

    # ------------------------------------------------------------------
    def _generate_queries(self, question: str) -> list[dict]:
        prompt = f"Research question: {question}"
        messages = [SystemMessage(content=_SEARCH_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, dict):
                return parsed.get("queries", [{"query": question, "type": "perplexity"}])[:3]
        except Exception:
            pass
        return [{"query": question, "type": "perplexity"}]

    def _identify_paths(
        self, topic: str, questions: list[str], search_results: list[str]
    ) -> list[dict]:
        combined = "\n---\n".join(search_results)[:8000]
        prompt = (
            f"Research topic: {topic}\n"
            f"Questions: {json.dumps(questions)}\n\n"
            f"Search results:\n{combined}\n\n"
            "Identify 2-5 distinct research paths."
        )
        messages = [SystemMessage(content=_PATH_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, list):
                # Ensure IDs
                for i, p in enumerate(parsed):
                    if "id" not in p:
                        p["id"] = f"path-{i+1}"
                    p.setdefault("status", "proposed")
                return parsed
        except Exception:
            pass
        # Fallback: generate simple paths from the topic
        return [
            {"id": "path-1", "title": f"Theoretical foundations of {topic[:40]}",
             "description": "Explore the theoretical underpinnings.", "status": "proposed"},
            {"id": "path-2", "title": f"Practical applications of {topic[:40]}",
             "description": "Survey real-world applications and case studies.", "status": "proposed"},
            {"id": "path-3", "title": f"Recent advances in {topic[:40]}",
             "description": "Review state-of-the-art developments.", "status": "proposed"},
        ]
