"""Tests for QuotaManager."""

import sys
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import AppConfig, load_config
from utils.quota_manager import QuotaManager


def _make_config() -> AppConfig:
    return load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))


def test_acquire_success():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    assert qm.acquire("perplexity_search", "broad_survey") is True


def test_acquire_exhausts_quota():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    limit = cfg.quotas.broad_survey.perplexity_search
    for _ in range(limit):
        assert qm.acquire("perplexity_search", "broad_survey") is True
    assert qm.acquire("perplexity_search", "broad_survey") is False


def test_get_remaining():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    initial = qm.get_remaining("perplexity_search", "broad_survey")
    qm.acquire("perplexity_search", "broad_survey")
    assert qm.get_remaining("perplexity_search", "broad_survey") == initial - 1


def test_reset():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    limit = cfg.quotas.broad_survey.perplexity_search
    for _ in range(limit):
        qm.acquire("perplexity_search", "broad_survey")
    assert qm.get_remaining("perplexity_search", "broad_survey") == 0
    qm.reset(cfg)
    assert qm.get_remaining("perplexity_search", "broad_survey") == limit


def test_unknown_stage_always_allows():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    assert qm.acquire("perplexity_search", "nonexistent_stage") is True


def test_unknown_tool_always_allows():
    cfg = _make_config()
    qm = QuotaManager(cfg)
    assert qm.acquire("nonexistent_tool", "broad_survey") is True
