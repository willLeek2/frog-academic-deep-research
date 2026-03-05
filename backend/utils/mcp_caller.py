"""MCP call wrapper with quota checking, logging, paper registration, retry,
and content truncation.

Calls the underlying APIs directly via *httpx* synchronous client.  If the
MCP Node.js server is available, users can swap in the ``mcp`` SDK transport
instead – the interface is identical.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import httpx

from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger
from utils.token_counter import count_tokens

# Maximum tokens before truncating fetched content
_MAX_CONTENT_TOKENS = 6000
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


def _truncate(text: str, max_tokens: int = _MAX_CONTENT_TOKENS) -> str:
    """Truncate *text* if it exceeds *max_tokens*."""
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text
    # Rough char-based truncation (avg ~4 chars/token for English)
    ratio = max_tokens / tokens
    cut = int(len(text) * ratio)
    return text[:cut] + "\n\n[TRUNCATED — original exceeded token limit]"


def _retry_request(fn, *args, **kwargs):
    """Execute *fn* with exponential-backoff retry."""
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    raise last_exc  # type: ignore[misc]


class MCPCaller:
    """Wraps MCP calls with automatic quota management, logging, paper
    registration, retry and truncation."""

    def __init__(
        self,
        quota_manager: QuotaManager,
        run_logger: RunLogger,
        paper_registry: PaperRegistry,
    ) -> None:
        self.quota_manager = quota_manager
        self.run_logger = run_logger
        self.paper_registry = paper_registry
        self._http = httpx.Client(timeout=60.0)
        self._perplexity_key = os.getenv("PERPLEXITY_API_KEY", "")
        self._jina_key = os.getenv("JINA_API_KEY", "")

    # ------------------------------------------------------------------
    # Perplexity search
    # ------------------------------------------------------------------
    def perplexity_search(
        self, query: str, stage: str, mode: str = "academic"
    ) -> str:
        """Search via Perplexity API.  *mode* can be ``academic`` or ``web``."""
        if not self.quota_manager.acquire("perplexity_search", stage):
            return "[QUOTA EXHAUSTED] perplexity_search quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="perplexity_search",
            detail={"query": query, "mode": mode},
        )

        if not self._perplexity_key:
            return self._mock_perplexity(query)

        model = "sonar" if mode == "web" else "sonar"
        try:
            resp = _retry_request(
                self._http.post,
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._perplexity_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            citations = data.get("citations", [])
            # Register papers from citations
            for url in citations:
                pid = f"paper-{uuid.uuid4().hex[:8]}"
                self.paper_registry.register(pid, {"source": "perplexity", "url": url, "query": query})
            result_text = _truncate(content)
            if citations:
                result_text += "\n\nReferences:\n" + "\n".join(f"- {c}" for c in citations)
            return result_text
        except Exception as exc:
            self.run_logger.log(stage=stage, agent="mcp_caller", event_type="error",
                                detail={"tool": "perplexity_search", "error": str(exc)})
            return f"[ERROR] Perplexity search failed: {exc}"

    # ------------------------------------------------------------------
    # Jina fetch (crawl)
    # ------------------------------------------------------------------
    def jina_fetch(self, url: str, stage: str) -> str:
        """Fetch URL content via Jina Reader API."""
        if not self.quota_manager.acquire("jina_fetch", stage):
            return "[QUOTA EXHAUSTED] jina_fetch quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="jina_fetch",
            detail={"url": url},
        )

        if not self._jina_key:
            return self._mock_jina_fetch(url)

        try:
            headers = {
                "Authorization": f"Bearer {self._jina_key}",
                "Accept": "text/markdown",
            }
            resp = _retry_request(
                self._http.get,
                f"https://r.jina.ai/{url}",
                headers=headers,
            )
            resp.raise_for_status()
            content = resp.text
            pid = f"paper-{uuid.uuid4().hex[:8]}"
            self.paper_registry.register(pid, {"source": "jina_fetch", "url": url})
            return _truncate(content)
        except Exception as exc:
            self.run_logger.log(stage=stage, agent="mcp_caller", event_type="error",
                                detail={"tool": "jina_fetch", "error": str(exc)})
            return f"[ERROR] Jina fetch failed: {exc}"

    # ------------------------------------------------------------------
    # Web / Jina search
    # ------------------------------------------------------------------
    def web_search(self, query: str, stage: str) -> str:
        """General web search via Jina Search API."""
        if not self.quota_manager.acquire("web_search", stage):
            return "[QUOTA EXHAUSTED] web_search quota for this stage has been used up."

        self.run_logger.log(
            stage=stage,
            agent="mcp_caller",
            event_type="web_search",
            detail={"query": query},
        )

        if not self._jina_key:
            return self._mock_web_search(query)

        try:
            headers = {
                "Authorization": f"Bearer {self._jina_key}",
                "Accept": "application/json",
            }
            resp = _retry_request(
                self._http.get,
                f"https://s.jina.ai/{query}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text}
            results = data.get("data", []) if isinstance(data, dict) else []
            parts: list[str] = []
            for item in results[:10]:
                title = item.get("title", "")
                link = item.get("url", "")
                snippet = item.get("description", item.get("content", ""))[:300]
                parts.append(f"- [{title}]({link}): {snippet}")
            return "\n".join(parts) if parts else _truncate(resp.text)
        except Exception as exc:
            self.run_logger.log(stage=stage, agent="mcp_caller", event_type="error",
                                detail={"tool": "web_search", "error": str(exc)})
            return f"[ERROR] Web search failed: {exc}"

    # ------------------------------------------------------------------
    # Mock fallbacks (used when API keys are absent)
    # ------------------------------------------------------------------
    def _mock_perplexity(self, query: str) -> str:
        pid = f"paper-{uuid.uuid4().hex[:8]}"
        self.paper_registry.register(pid, {"title": f"Mock Paper: {query[:50]}", "source": "perplexity_search", "url": f"https://example.com/paper/{pid}"})
        return (
            f"[Mock Perplexity Result] Found 3 relevant papers for '{query}':\n"
            f"1. 'Advances in {query}' (2025) - https://example.com/paper/{pid}\n"
            f"2. 'A Survey of {query}' (2024) - https://example.com/paper/survey-001\n"
            f"3. 'New Methods in {query}' (2025) - https://example.com/paper/methods-001\n"
        )

    def _mock_jina_fetch(self, url: str) -> str:
        return (
            f"[Mock Jina Fetch] Content from {url}:\n"
            "Abstract: This paper presents a novel approach to the problem...\n"
            "Keywords: deep learning, research, survey\n"
            "Content length: ~2000 words (mock)"
        )

    def _mock_web_search(self, query: str) -> str:
        return (
            f"[Mock Web Search] Results for '{query}':\n"
            "1. https://example.com/result1 - Relevant page about the topic\n"
            "2. https://example.com/result2 - Another relevant source\n"
            "3. https://example.com/result3 - Additional reference\n"
        )
