"""MCP call wrapper with quota checking, logging, and paper registration.

This module provides a unified interface for calling MCP tools. In Issue 1,
all MCP calls return **mock data**. Real MCP integration will be added in
Issue 2.
"""

from __future__ import annotations

import uuid
from typing import Optional

from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger


class MCPCaller:
    """Wraps MCP calls with automatic quota management, logging, and paper
    registration."""

    def __init__(
        self,
        quota_manager: QuotaManager,
        run_logger: RunLogger,
        paper_registry: PaperRegistry,
    ) -> None:
        self.quota_manager = quota_manager
        self.run_logger = run_logger
        self.paper_registry = paper_registry

    def perplexity_search(self, query: str, stage: str) -> str:
        """Search via Perplexity MCP (mock implementation)."""
        if not self.quota_manager.acquire("perplexity_search", stage):
            return "[QUOTA EXHAUSTED] perplexity_search quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="perplexity_search",
            detail={"query": query},
        )

        # Mock results
        paper_id = f"paper-{uuid.uuid4().hex[:8]}"
        self.paper_registry.register(
            paper_id,
            {
                "title": f"Mock Paper: {query[:50]}",
                "source": "perplexity_search",
                "url": f"https://example.com/paper/{paper_id}",
            },
        )

        return (
            f"[Mock Perplexity Result] Found 3 relevant papers for '{query}':\n"
            f"1. 'Advances in {query}' (2025) - https://example.com/paper/{paper_id}\n"
            f"2. 'A Survey of {query}' (2024) - https://example.com/paper/survey-001\n"
            f"3. 'New Methods in {query}' (2025) - https://example.com/paper/methods-001\n"
        )

    def jina_fetch(self, url: str, stage: str) -> str:
        """Fetch URL content via Jina MCP (mock implementation)."""
        if not self.quota_manager.acquire("jina_fetch", stage):
            return "[QUOTA EXHAUSTED] jina_fetch quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="jina_fetch",
            detail={"url": url},
        )

        return (
            f"[Mock Jina Fetch] Content from {url}:\n"
            "Abstract: This paper presents a novel approach to the problem...\n"
            "Keywords: deep learning, research, survey\n"
            "Content length: ~2000 words (mock)"
        )

    def web_search(self, query: str, stage: str) -> str:
        """General web search via MCP (mock implementation)."""
        if not self.quota_manager.acquire("web_search", stage):
            return "[QUOTA EXHAUSTED] web_search quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="web_search",
            detail={"query": query},
        )

        return (
            f"[Mock Web Search] Results for '{query}':\n"
            "1. https://example.com/result1 - Relevant page about the topic\n"
            "2. https://example.com/result2 - Another relevant source\n"
            "3. https://example.com/result3 - Additional reference\n"
        )
