"""Three-level summary generator with file-system caching.

Levels
------
- **L1**: 1–2 sentence overview
- **L2**: One paragraph (~150 words)
- **L3**: Detailed multi-paragraph summary with structure
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_helpers import extract_text

_SYSTEM = "You are a precise academic summariser."

_LEVEL_PROMPTS = {
    "L1": "Summarise the following content in 1–2 sentences:\n\n{content}",
    "L2": "Write a single-paragraph (~150 words) summary of the following content:\n\n{content}",
    "L3": (
        "Write a detailed, structured summary (multiple paragraphs, use "
        "sub-headings where appropriate) of the following content:\n\n{content}"
    ),
}


class SummaryGenerator:
    """Generate and cache L1/L2/L3 summaries."""

    def __init__(self, llm: BaseChatModel, cache_dir: str | Path) -> None:
        self._llm = llm
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _cache_key(self, content: str, level: str, focus: Optional[str]) -> str:
        blob = f"{level}|{focus or ''}|{content}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _read_cache(self, key: str) -> Optional[str]:
        p = self._cache_dir / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _write_cache(self, key: str, summary: str) -> None:
        p = self._cache_dir / f"{key}.txt"
        p.write_text(summary, encoding="utf-8")

    # ------------------------------------------------------------------
    def generate(
        self,
        content: str,
        level: str = "L1",
        focus: Optional[str] = None,
    ) -> str:
        """Return a summary at the requested *level*.

        Args:
            content: Source text.
            level: ``L1``, ``L2``, or ``L3``.
            focus: Optional focus hint appended to the prompt.
        """
        if level not in _LEVEL_PROMPTS:
            level = "L1"

        key = self._cache_key(content, level, focus)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        user_prompt = _LEVEL_PROMPTS[level].format(content=content[:8000])
        if focus:
            user_prompt += f"\n\nFocus on: {focus}"

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt),
        ]
        resp = self._llm.invoke(messages)
        summary = extract_text(resp)
        self._write_cache(key, summary)
        return summary
