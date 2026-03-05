"""Stage 4: Deep researcher agent.

Performs DFS-style deep research on *valuable* paths and brief research on
*suboptimal* paths.  Tracks citation chains, relevance decay, and quota
consumption.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import ResearchState
from utils.llm_helpers import extract_text, parse_json_response
from utils.mcp_caller import MCPCaller
from utils.relevance_evaluator import RelevanceEvaluator
from utils.run_logger import RunLogger
from utils.stop_controller import StopController
from utils.summary_generator import SummaryGenerator

_MAX_DFS_DEPTH = 3
_RELEVANCE_THRESHOLD = 4  # Below this, stop following
_CONFIDENCE_DECAY_THRESHOLD = 0.4  # If confidence drops below, trigger downgrade

_REF_EXTRACT_SYSTEM = (
    "You are an academic reference extractor. Given a paper's content, "
    "extract the most relevant references that should be followed for "
    "deeper investigation. Output ONLY a JSON object:\n"
    '{"references": [{"title": "<title>", "url": "<url if available>", '
    '"relevance_hint": "<why relevant>"}], '
    '"path_confidence": <0.0-1.0>, '
    '"new_path_proposal": null | {"title": "...", "description": "..."}}'
)


class DeepResearcherAgent:
    """Callable LangGraph node: DFS deep research."""

    def __init__(
        self,
        heavy_llm: BaseChatModel,
        light_llm: BaseChatModel,
        mcp_caller: MCPCaller,
        summary_gen: SummaryGenerator,
        relevance_eval: RelevanceEvaluator,
        run_logger: RunLogger,
        stop_controller: StopController,
        run_dir: str | Path,
    ) -> None:
        self._heavy = heavy_llm
        self._light = light_llm
        self._mcp = mcp_caller
        self._sg = summary_gen
        self._rel = relevance_eval
        self._logger = run_logger
        self._stop = stop_controller
        self._run_dir = Path(run_dir)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        paths = state.get("research_paths", [])
        evaluations = state.get("path_evaluations", [])
        ctx = state.get("extracted_context") or {}
        topic = ctx.get("topic", "unknown")

        eval_map = {e["path_id"]: e for e in evaluations}

        notes: dict[str, str] = dict(state.get("research_notes", {}))
        paper_ids: list[str] = list(state.get("paper_ids", []))
        status_changes: list[dict] = list(state.get("path_status_changes", []))
        new_proposals: list[dict] = list(state.get("new_path_proposals", []))

        stage = "deep_research"

        for p in paths:
            if self._stop.is_stop_requested():
                break
            pid = p["id"]
            ev = eval_map.get(pid, {})
            category = ev.get("category", "suboptimal")

            if category == "valuable":
                note, pids, sc, np = self._deep_research_path(topic, p, stage)
                notes[pid] = note
                paper_ids.extend(pids)
                status_changes.extend(sc)
                new_proposals.extend(np)
            else:
                note = self._brief_research_path(topic, p, stage)
                notes[pid] = note

        return {
            "research_notes": notes,
            "paper_ids": paper_ids,
            "path_status_changes": status_changes,
            "new_path_proposals": new_proposals,
            "current_stage": "deep_research_done",
        }

    # ------------------------------------------------------------------
    # Valuable path: DFS
    # ------------------------------------------------------------------
    def _deep_research_path(
        self, topic: str, path: dict, stage: str
    ) -> tuple[str, list[str], list[dict], list[dict]]:
        pid = path["id"]
        pdir = self._run_dir / "paths" / pid
        pdir.mkdir(parents=True, exist_ok=True)

        notes_parts: list[str] = []
        paper_ids: list[str] = []
        status_changes: list[dict] = []
        new_proposals: list[dict] = []
        confidence_history: list[float] = []

        # Use entry papers or search for initial content
        entry_papers = path.get("entry_papers", [])
        if not entry_papers:
            # Search for entry content
            result = self._mcp.perplexity_search(
                f"{topic} {path.get('title', '')}", stage
            )
            notes_parts.append(f"## Initial survey for {path.get('title', pid)}\n\n{result}")
            urls = re.findall(r"https?://[^\s\)]+", result)
            entry_papers = urls[:2]

        # DFS stack: (url_or_title, depth)
        stack: list[tuple[str, int]] = [(e, 0) for e in entry_papers[:3]]
        visited: set[str] = set()

        while stack and not self._stop.is_stop_requested():
            item, depth = stack.pop()
            if item in visited or depth > _MAX_DFS_DEPTH:
                continue
            visited.add(item)

            # Fetch content
            if item.startswith("http"):
                content = self._mcp.jina_fetch(item, stage)
            else:
                content = self._mcp.perplexity_search(
                    f"{item} {topic}", stage
                )

            if content.startswith("[QUOTA EXHAUSTED]") or content.startswith("[ERROR]"):
                notes_parts.append(f"*Quota/error for {item}*: {content[:200]}")
                continue

            # Generate summary (L3 for entry, L2 for deeper)
            level = "L3" if depth == 0 else "L2"
            summary = self._sg.generate(content, level=level, focus=topic)

            # Save raw + summary
            raw_dir = self._run_dir / "papers" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^\w\-.]", "_", item)[:80]
            (raw_dir / f"{safe_name}.md").write_text(content[:20000], encoding="utf-8")

            sum_dir = self._run_dir / "papers" / "summaries"
            sum_dir.mkdir(parents=True, exist_ok=True)
            (sum_dir / f"{safe_name}_{level}.md").write_text(summary, encoding="utf-8")

            paper_ids.append(safe_name)
            notes_parts.append(f"### [{level}] {item} (depth={depth})\n\n{summary}")

            # Evaluate relevance
            rel = self._rel.evaluate(topic, summary)
            score = rel.get("score", 5)

            self._logger.log(stage, "deep_researcher", "paper_processed", {
                "path_id": pid, "item": item[:100], "depth": depth,
                "relevance": score, "summary_level": level,
            })

            if score < _RELEVANCE_THRESHOLD:
                notes_parts.append(f"*Relevance too low ({score}/10) — stopping this branch.*")
                continue

            # Extract references and path confidence
            refs_data = self._extract_references(content, topic)
            confidence = refs_data.get("path_confidence", 0.7)
            confidence_history.append(confidence)

            # Check for path status change
            if len(confidence_history) >= 2 and all(
                c < _CONFIDENCE_DECAY_THRESHOLD for c in confidence_history[-2:]
            ):
                status_changes.append({
                    "path_id": pid,
                    "change": "downgrade",
                    "reason": "Consecutive low confidence scores",
                    "confidence_history": confidence_history[-3:],
                })

            # Check for new path proposal
            proposal = refs_data.get("new_path_proposal")
            if proposal and isinstance(proposal, dict) and proposal.get("title"):
                new_proposals.append({
                    "source_path_id": pid,
                    "title": proposal["title"],
                    "description": proposal.get("description", ""),
                })

            # Add references to DFS stack
            for ref in refs_data.get("references", [])[:3]:
                ref_url = ref.get("url", "")
                ref_title = ref.get("title", "")
                target = ref_url if ref_url.startswith("http") else ref_title
                if target and target not in visited:
                    stack.append((target, depth + 1))

        # Compile notes
        full_notes = "\n\n".join(notes_parts)
        (pdir / "research_notes.md").write_text(full_notes, encoding="utf-8")

        return full_notes, paper_ids, status_changes, new_proposals

    # ------------------------------------------------------------------
    # Suboptimal path: brief
    # ------------------------------------------------------------------
    def _brief_research_path(self, topic: str, path: dict, stage: str) -> str:
        pid = path["id"]
        pdir = self._run_dir / "paths" / pid
        pdir.mkdir(parents=True, exist_ok=True)

        result = self._mcp.perplexity_search(
            f"{topic} {path.get('title', '')} overview", stage
        )
        summary = self._sg.generate(result, level="L1", focus=topic)

        note = (
            f"## Brief survey: {path.get('title', pid)}\n\n"
            f"{summary}\n\n"
            f"*This path was classified as suboptimal and received only a brief survey.*"
        )
        (pdir / "research_notes.md").write_text(note, encoding="utf-8")

        self._logger.log(stage, "deep_researcher", "brief_survey", {"path_id": pid})
        return note

    # ------------------------------------------------------------------
    # Helper: extract references
    # ------------------------------------------------------------------
    def _extract_references(self, content: str, topic: str) -> dict:
        prompt = (
            f"Research topic: {topic}\n\n"
            f"Paper content (truncated):\n{content[:4000]}\n\n"
            "Extract the most relevant references, assess path confidence, "
            "and propose any new research direction if warranted."
        )
        messages = [
            SystemMessage(content=_REF_EXTRACT_SYSTEM),
            HumanMessage(content=prompt),
        ]
        try:
            resp = self._heavy.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"references": [], "path_confidence": 0.7, "new_path_proposal": None}
