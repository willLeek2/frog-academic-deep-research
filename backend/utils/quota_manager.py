"""Quota manager for controlling MCP call limits per stage."""

from __future__ import annotations

import threading
from typing import Optional

from core.config_loader import AppConfig, StageQuota


class QuotaManager:
    """Thread-safe quota manager for MCP tool calls.

    Tracks remaining call counts per tool per stage and prevents exceeding
    configured limits.
    """

    def __init__(self, config: AppConfig) -> None:
        self._lock = threading.Lock()
        # Build {stage: {tool_name: remaining}} from config
        self._remaining: dict[str, dict[str, int]] = {}
        for stage_name in ("broad_survey", "deep_research", "writing"):
            stage_quota: Optional[StageQuota] = getattr(
                config.quotas, stage_name, None
            )
            if stage_quota:
                self._remaining[stage_name] = {
                    "perplexity_search": stage_quota.perplexity_search,
                    "web_search": stage_quota.web_search,
                    "jina_fetch": stage_quota.jina_fetch,
                }

    def acquire(self, tool_name: str, stage: str) -> bool:
        """Try to acquire one quota unit for *tool_name* in *stage*.

        Returns True if quota was available and consumed, False otherwise.
        """
        with self._lock:
            stage_quotas = self._remaining.get(stage)
            if stage_quotas is None:
                return True  # No quota configured for this stage
            remaining = stage_quotas.get(tool_name)
            if remaining is None:
                return True  # No quota configured for this tool
            if remaining <= 0:
                return False
            stage_quotas[tool_name] = remaining - 1
            return True

    def get_remaining(self, tool_name: str, stage: str) -> int:
        """Return the number of remaining quota units."""
        with self._lock:
            stage_quotas = self._remaining.get(stage, {})
            return stage_quotas.get(tool_name, 0)

    def reset(self, config: AppConfig) -> None:
        """Reset all quotas from config."""
        with self._lock:
            for stage_name in ("broad_survey", "deep_research", "writing"):
                stage_quota: Optional[StageQuota] = getattr(
                    config.quotas, stage_name, None
                )
                if stage_quota:
                    self._remaining[stage_name] = {
                        "perplexity_search": stage_quota.perplexity_search,
                        "web_search": stage_quota.web_search,
                        "jina_fetch": stage_quota.jina_fetch,
                    }
