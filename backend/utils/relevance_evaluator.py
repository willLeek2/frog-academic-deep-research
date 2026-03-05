"""Relevance evaluator – scores content against a research question.

Uses the *light* LLM to return a 0-10 relevance score.
"""

from __future__ import annotations

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_helpers import extract_text

_SYSTEM = (
    "You are an academic relevance evaluator. Given a research question and "
    "a summary of a paper/content, output ONLY a JSON object: "
    '{"score": <0-10>, "reason": "<one sentence>"}'
)


class RelevanceEvaluator:
    """Score content relevance against a research question using a light LLM."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    def evaluate(self, question: str, summary: str) -> dict:
        """Return ``{"score": int, "reason": str}``."""
        prompt = (
            f"Research question: {question}\n\n"
            f"Content summary: {summary}\n\n"
            "Rate relevance 0-10 and provide a one-sentence reason."
        )
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ]
        resp = self._llm.invoke(messages)
        text = extract_text(resp)
        return self._parse(text)

    @staticmethod
    def _parse(text: str) -> dict:
        """Best-effort JSON extraction."""
        # Try to find JSON in the response
        match = re.search(r"\{[^}]+\}", text)
        if match:
            import json
            try:
                data = json.loads(match.group())
                score = max(0, min(10, int(data.get("score", 5))))
                return {"score": score, "reason": str(data.get("reason", ""))}
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback: look for a bare number
        nums = re.findall(r"\b(\d+)\b", text)
        if nums:
            score = max(0, min(10, int(nums[0])))
            return {"score": score, "reason": text.strip()[:200]}
        return {"score": 5, "reason": "Could not parse relevance score."}
