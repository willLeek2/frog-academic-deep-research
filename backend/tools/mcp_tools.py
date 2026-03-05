"""LangChain tool definitions wrapping MCP calls.

These tools are bound to the LLM for tool-calling. In Issue 1, they return
mock data via the MCPCaller.
"""

from __future__ import annotations

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Module-level MCPCaller instance, set at startup by main.py
# ---------------------------------------------------------------------------
_mcp_caller = None
_current_stage = "broad_survey"


def set_mcp_caller(caller) -> None:
    """Inject the MCPCaller instance at startup."""
    global _mcp_caller
    _mcp_caller = caller


def set_current_stage(stage: str) -> None:
    """Update the stage used for quota tracking."""
    global _current_stage
    _current_stage = stage


@tool
def perplexity_search(query: str) -> str:
    """Search for academic papers using Perplexity. Returns an LLM summary
    with reference links."""
    if _mcp_caller is None:
        return "[ERROR] MCP caller not initialized"
    return _mcp_caller.perplexity_search(query, _current_stage)


@tool
def jina_fetch(url: str) -> str:
    """Fetch and extract content from a URL using Jina."""
    if _mcp_caller is None:
        return "[ERROR] MCP caller not initialized"
    return _mcp_caller.jina_fetch(url, _current_stage)


@tool
def web_search(query: str) -> str:
    """Perform a general web search."""
    if _mcp_caller is None:
        return "[ERROR] MCP caller not initialized"
    return _mcp_caller.web_search(query, _current_stage)
