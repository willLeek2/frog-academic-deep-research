"""Small helpers for interacting with LangChain LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_text(message) -> str:
    """Return the text content from an AIMessage (or similar)."""
    if hasattr(message, "content"):
        return str(message.content)
    return str(message)


def parse_json_response(text: str) -> Any:
    """Best-effort extraction of a JSON object/array from LLM output.

    Handles common cases:
    - raw JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON preceded/followed by prose
    """
    # Strip markdown fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the first { ... } or [ ... ] block
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    return None
