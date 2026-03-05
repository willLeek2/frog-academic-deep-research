"""Tests for utils/mcp_caller.py – mock fallback paths."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import load_config
from utils.mcp_caller import MCPCaller, _truncate
from utils.paper_registry import PaperRegistry
from utils.quota_manager import QuotaManager
from utils.run_logger import RunLogger


def _make_caller(tmpdir: str) -> MCPCaller:
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    logger = RunLogger(Path(tmpdir) / "log.jsonl")
    registry = PaperRegistry()
    qm = QuotaManager(cfg)
    # No API keys set → falls back to mock
    caller = MCPCaller(qm, logger, registry)
    return caller


def test_truncate_short():
    assert _truncate("short text", max_tokens=100) == "short text"


def test_truncate_long():
    long_text = "word " * 10000
    result = _truncate(long_text, max_tokens=100)
    assert "[TRUNCATED" in result
    assert len(result) < len(long_text)


def test_perplexity_search_mock():
    with tempfile.TemporaryDirectory() as td:
        caller = _make_caller(td)
        result = caller.perplexity_search("test query", "broad_survey")
        assert "Mock Perplexity Result" in result or "test query" in result


def test_jina_fetch_mock():
    with tempfile.TemporaryDirectory() as td:
        caller = _make_caller(td)
        result = caller.jina_fetch("https://example.com", "broad_survey")
        assert "Mock Jina Fetch" in result or "example.com" in result


def test_web_search_mock():
    with tempfile.TemporaryDirectory() as td:
        caller = _make_caller(td)
        result = caller.web_search("test query", "broad_survey")
        assert "Mock Web Search" in result or "test query" in result


def test_quota_exhaustion():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    with tempfile.TemporaryDirectory() as td:
        logger = RunLogger(Path(td) / "log.jsonl")
        registry = PaperRegistry()
        qm = QuotaManager(cfg)
        caller = MCPCaller(qm, logger, registry)
        # Exhaust perplexity_search quota for broad_survey
        limit = cfg.quotas.broad_survey.perplexity_search
        for _ in range(limit):
            caller.perplexity_search("q", "broad_survey")
        result = caller.perplexity_search("q", "broad_survey")
        assert "QUOTA EXHAUSTED" in result
