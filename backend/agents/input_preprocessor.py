"""Stage 1: Input preprocessor agent.

Reads the raw research-requirement markdown document and uses the heavy LLM
to extract structured context (topic, questions, keywords, constraints).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.state import ResearchState
from utils.llm_helpers import extract_text, parse_json_response

_SYSTEM = (
    "You are a research requirement analyst. Given a research requirement "
    "document, extract structured information. Output ONLY a JSON object with "
    "these keys:\n"
    '  "topic": "<main research topic>",\n'
    '  "questions": ["<specific research question 1>", ...],\n'
    '  "keywords": ["<keyword1>", ...],\n'
    '  "constraints": "<any constraints or scope limitations>",\n'
    '  "expected_scope": "<expected scope of the research>"\n'
)


class InputPreprocessor:
    """Callable LangGraph node: extract structured context from raw input."""

    def __init__(self, llm: BaseChatModel, run_dir: str | Path) -> None:
        self._llm = llm
        self._run_dir = Path(run_dir)

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        raw_path = state.get("raw_input_path", "")
        content = ""
        if raw_path and Path(raw_path).exists():
            content = Path(raw_path).read_text(encoding="utf-8")

        if not content:
            extracted = {
                "topic": "Unknown topic",
                "questions": [],
                "keywords": [],
                "constraints": "",
                "expected_scope": "",
                "raw_length": 0,
            }
            return {"extracted_context": extracted, "current_stage": "input_preprocessing_done"}

        prompt = f"Research requirement document:\n\n{content[:6000]}"
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]

        try:
            resp = self._llm.invoke(messages)
            text = extract_text(resp)
            parsed = parse_json_response(text)
            if parsed and isinstance(parsed, dict):
                extracted = {
                    "topic": parsed.get("topic", content[:200]),
                    "questions": parsed.get("questions", []),
                    "keywords": parsed.get("keywords", []),
                    "constraints": parsed.get("constraints", ""),
                    "expected_scope": parsed.get("expected_scope", ""),
                    "raw_length": len(content),
                }
            else:
                raise ValueError("LLM did not return valid JSON")
        except Exception:
            # Fallback: simple extraction without LLM
            extracted = {
                "topic": content[:200],
                "questions": [],
                "keywords": ["research", "survey"],
                "constraints": "",
                "expected_scope": "",
                "raw_length": len(content),
            }

        # Persist extracted context
        out = self._run_dir / "extracted_context.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"extracted_context": extracted, "current_stage": "input_preprocessing_done"}
