"""Stage 5C/5D: Writer agent with supplement-research support.

Writes each section using its context pack.  Marks terminology with
``[[TERM:XXX]]`` and can request supplementary research when information is
insufficient.
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
from utils.run_logger import RunLogger
from utils.stop_controller import StopController

_MAX_SUPPLEMENT_PER_SECTION = 2

_WRITE_SYSTEM = (
    "You are an academic report writer. Write the requested section based on "
    "the provided context. Guidelines:\n"
    "- Write in a clear, academic style\n"
    "- When you use a technical term for the first time, mark it as "
    "[[TERM:term_name]]\n"
    "- If you lack sufficient information for a claim, instead of fabricating, "
    'output a JSON block: {"supplement_request": {"query": "<what to search>", '
    '"reason": "<why needed>"}}\n'
    "- Include inline citations where possible using [Author, Year] format\n"
    "- Target approximately the word count specified\n"
)


class WriterAgent:
    """Callable LangGraph node: sequential section writing."""

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
        outline = state.get("outline") or {}
        sections = outline.get("sections", [])
        context_packs = state.get("context_packs", {})
        drafts: dict[str, str] = dict(state.get("drafts", {}))
        terminology: list[str] = list(state.get("terminology", []))
        supplement_requests: dict[str, int] = dict(state.get("supplement_requests", {}))

        drafts_dir = self._run_dir / "writing" / "drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)

        stage = "writing"

        for sec in sections:
            if self._stop.is_stop_requested():
                break

            sid = sec["id"]
            pack = context_packs.get(sid, {})
            context_text = pack.get("context", "")

            draft, supplement = self._write_section(sec, context_text, stage)

            # Handle supplement research (up to MAX retries)
            attempts = supplement_requests.get(sid, 0)
            while supplement and attempts < _MAX_SUPPLEMENT_PER_SECTION:
                if self._stop.is_stop_requested():
                    break
                attempts += 1
                extra = self._do_supplement(supplement, stage)
                context_text += f"\n\n### Supplement research #{attempts}\n{extra}"
                draft, supplement = self._write_section(sec, context_text, stage)

            supplement_requests[sid] = attempts
            drafts[sid] = draft

            # Extract terminology
            terms = re.findall(r"\[\[TERM:([^\]]+)\]\]", draft)
            terminology.extend(t for t in terms if t not in terminology)

            # Save draft
            (drafts_dir / f"{sid}.md").write_text(draft, encoding="utf-8")

            self._logger.log(stage, "writer", "section_written", {
                "section_id": sid, "words": len(draft.split()),
                "supplements": attempts, "terms_found": len(terms),
            })

        return {
            "drafts": drafts,
            "terminology": terminology,
            "supplement_requests": supplement_requests,
            "current_stage": "writing_done",
        }

    # ------------------------------------------------------------------
    def _write_section(
        self, section: dict, context: str, stage: str
    ) -> tuple[str, dict | None]:
        """Write a section.  Returns ``(draft_text, supplement_request_or_None)``."""
        title = section.get("title", section["id"])
        target_words = section.get("target_words", 800)
        prompt = (
            f"Section: {title}\n"
            f"Target word count: ~{target_words}\n"
            f"Description: {section.get('description', '')}\n\n"
            f"Context:\n{context[:8000]}\n\n"
            "Write this section now."
        )
        messages = [SystemMessage(content=_WRITE_SYSTEM), HumanMessage(content=prompt)]
        try:
            resp = self._llm.invoke(messages)
            text = extract_text(resp)
        except Exception as exc:
            text = f"[Writing error: {exc}]"

        # Check for supplement request in the output
        supplement = None
        parsed = parse_json_response(text)
        if parsed and isinstance(parsed, dict) and "supplement_request" in parsed:
            supplement = parsed["supplement_request"]

        return text, supplement

    # ------------------------------------------------------------------
    def _do_supplement(self, request: dict, stage: str) -> str:
        query = request.get("query", "")
        if not query:
            return ""
        result = self._mcp.perplexity_search(query, stage)
        self._logger.log(stage, "writer", "supplement_research", {"query": query})
        return result
