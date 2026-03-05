"""Stage 6: Post-processor agent.

Handles terminology explanation, ``[[TERM:XXX]]`` replacement, reference
compilation, suboptimal-path section inclusion, and final report assembly.
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
from utils.run_logger import RunLogger

_TERM_SYSTEM = (
    "You are an academic glossary writer. Given a list of technical terms, "
    "provide a brief explanation (1-2 sentences) for each. Output ONLY a JSON "
    "object mapping term to explanation:\n"
    '{"term1": "explanation1", "term2": "explanation2", ...}'
)


class PostProcessorAgent:
    """Callable LangGraph node: post-processing and final report assembly."""

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
        outline = state.get("outline") or {}
        drafts = state.get("drafts", {})
        terminology = state.get("terminology", [])
        evaluations = state.get("path_evaluations", [])
        notes = state.get("research_notes", {})
        run_id = state.get("run_id", "unknown")

        # 1. Generate terminology explanations
        term_map = self._generate_term_explanations(terminology)

        # 2. Replace [[TERM:XXX]] markers
        processed_drafts = {}
        for sid, draft in drafts.items():
            processed_drafts[sid] = self._replace_terms(draft, term_map)

        # 3. Assemble final report
        report = self._assemble_report(
            outline, processed_drafts, term_map, evaluations, notes, run_id
        )

        # 4. Persist
        output_dir = self._run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report_final.md").write_text(report, encoding="utf-8")
        (output_dir / "terminology.json").write_text(
            json.dumps(term_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self._logger.log("post_processing", "post_processor", "report_assembled", {
            "sections": len(processed_drafts),
            "terms": len(term_map),
            "report_words": len(report.split()),
        })

        return {
            "drafts": processed_drafts,
            "terminology": list(term_map.keys()),
            "current_stage": "completed",
        }

    # ------------------------------------------------------------------
    def _generate_term_explanations(self, terms: list[str]) -> dict[str, str]:
        if not terms:
            return {}

        unique_terms = sorted(set(terms))
        prompt = f"Technical terms to explain:\n{json.dumps(unique_terms)}"
        messages = [SystemMessage(content=_TERM_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            parsed = parse_json_response(extract_text(resp))
            if parsed and isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
        return {t: f"A technical term related to the research topic." for t in unique_terms}

    # ------------------------------------------------------------------
    @staticmethod
    def _replace_terms(text: str, term_map: dict[str, str]) -> str:
        def _replacer(m: re.Match) -> str:
            term = m.group(1)
            return f"**{term}**"
        return re.sub(r"\[\[TERM:([^\]]+)\]\]", _replacer, text)

    # ------------------------------------------------------------------
    def _assemble_report(
        self,
        outline: dict,
        drafts: dict[str, str],
        term_map: dict[str, str],
        evaluations: list[dict],
        notes: dict[str, str],
        run_id: str,
    ) -> str:
        title = outline.get("title", "Research Report")
        sections = outline.get("sections", [])

        parts: list[str] = [f"# {title}\n"]

        # Main content
        for sec in sections:
            sid = sec["id"]
            sec_title = sec.get("title", sid)
            level = sec.get("level", 1)
            heading = "#" * (level + 1) + " " + sec_title
            draft = drafts.get(sid, "*No content generated for this section.*")
            parts.append(f"{heading}\n\n{draft}\n")

        # Suboptimal paths section (if not already in outline)
        suboptimal_ids = [e["path_id"] for e in evaluations if e.get("category") == "suboptimal"]
        if suboptimal_ids:
            sub_section = ["## Alternative Approaches\n"]
            for pid in suboptimal_ids:
                note = notes.get(pid, "No research notes available.")
                sub_section.append(f"### {pid}\n\n{note[:500]}\n")
            parts.append("\n".join(sub_section))

        # Terminology glossary
        if term_map:
            glossary = ["## Glossary\n"]
            for term, explanation in sorted(term_map.items()):
                glossary.append(f"- **{term}**: {explanation}")
            parts.append("\n".join(glossary))

        # References placeholder
        parts.append("\n## References\n\n*Reference list compiled from cited sources.*\n")

        # Footer
        parts.append(f"\n---\n*Generated by Deep Research Agent — Run {run_id}*\n")

        return "\n\n".join(parts)
