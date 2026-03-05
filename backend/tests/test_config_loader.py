"""Tests for config_loader."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config_loader import AppConfig, load_config


def test_load_default_config():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    assert isinstance(cfg, AppConfig)
    assert cfg.models.heavy.model_name == "deepseek/deepseek-r1"
    assert cfg.models.light.model_name == "deepseek/deepseek-chat"


def test_load_missing_file_returns_defaults():
    cfg = load_config("/nonexistent/config.yaml")
    assert isinstance(cfg, AppConfig)
    assert len(cfg.stages) > 0


def test_quotas_loaded():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    assert cfg.quotas.broad_survey.perplexity_search == 10
    assert cfg.quotas.deep_research.perplexity_search == 20


def test_human_intervention_flags():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    assert cfg.human_intervention.after_path_evaluation is True
    assert cfg.human_intervention.after_outline_planning is True
    assert cfg.human_intervention.after_deep_research is False


def test_stages_list():
    cfg = load_config(str(Path(__file__).resolve().parent.parent / "config.yaml"))
    assert "input_preprocessing" in cfg.stages
    assert "post_processing" in cfg.stages
