"""Context assembler for the writing stage (Stage 5B).

Gathers research notes, paper summaries, and adjacent-chapter context into a
single *context pack* for each section in the outline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.summary_generator import SummaryGenerator
from utils.token_counter import count_tokens

# Soft token budget per context pack (excluding the section prompt itself)
_TOKEN_BUDGET = 10000


class ContextAssembler:
    """Build per-section context packs from research artefacts."""

    def __init__(
        self,
        summary_generator: SummaryGenerator,
        run_dir: str | Path,
    ) -> None:
        self._sg = summary_generator
        self._run_dir = Path(run_dir)

    # ------------------------------------------------------------------
    def assemble(
        self,
        outline: dict,
        research_notes: dict[str, str],
        path_evaluations: list[dict],
        drafts: dict[str, str] | None = None,
    ) -> dict[str, dict]:
        """Return ``{section_id: context_pack}`` for every section."""
        sections = outline.get("sections", [])
        packs: dict[str, dict] = {}
        for idx, sec in enumerate(sections):
            sid = sec["id"]
            pack = self._build_pack(sec, idx, sections, research_notes, path_evaluations, drafts)
            packs[sid] = pack
        return packs

    # ------------------------------------------------------------------
    def _build_pack(
        self,
        section: dict,
        index: int,
        all_sections: list[dict],
        research_notes: dict[str, str],
        path_evaluations: list[dict],
        drafts: dict[str, str] | None,
    ) -> dict:
        parts: list[str] = []
        budget = _TOKEN_BUDGET

        # 1. Relevant research notes
        relevant_paths = section.get("related_paths", [])
        for pid in relevant_paths:
            note = research_notes.get(pid, "")
            if note:
                chunk = self._fit(note, budget)
                parts.append(f"### Research notes — {pid}\n{chunk}")
                budget -= count_tokens(chunk)
                if budget <= 0:
                    break

        # If no explicit related_paths, include all notes (trimmed)
        if not relevant_paths:
            for pid, note in research_notes.items():
                if budget <= 0:
                    break
                chunk = self._fit(note, min(budget, 2000))
                parts.append(f"### Research notes — {pid}\n{chunk}")
                budget -= count_tokens(chunk)

        # 2. Path evaluation summaries
        for ev in path_evaluations:
            if budget <= 0:
                break
            line = f"- Path {ev.get('path_id')}: score={ev.get('score')}, category={ev.get('category')}"
            parts.append(line)
            budget -= count_tokens(line)

        # 3. Adjacent chapter context (continuity)
        if drafts:
            # Previous chapter last paragraph
            if index > 0:
                prev_id = all_sections[index - 1]["id"]
                prev_draft = drafts.get(prev_id, "")
                if prev_draft:
                    tail = self._last_paragraph(prev_draft)
                    parts.append(f"### Previous chapter ending\n{tail}")

            # Next chapter brief
            if index < len(all_sections) - 1:
                nxt = all_sections[index + 1]
                parts.append(f"### Next chapter: {nxt.get('title', nxt['id'])}")

        context_text = "\n\n".join(parts)
        return {
            "section": section,
            "context": context_text,
            "token_count": count_tokens(context_text),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _fit(text: str, max_tokens: int) -> str:
        tokens = count_tokens(text)
        if tokens <= max_tokens:
            return text
        ratio = max_tokens / tokens
        cut = int(len(text) * ratio)
        return text[:cut] + "\n[...truncated]"

    @staticmethod
    def _last_paragraph(text: str) -> str:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paras[-1] if paras else ""
