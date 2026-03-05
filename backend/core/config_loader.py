"""Configuration loader for the deep research agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProviderRouting(BaseModel):
    order: list[str] = Field(default_factory=lambda: ["DeepInfra", "Fireworks"])
    allow_fallbacks: bool = True
    require_parameters: bool = True


class ModelConfig(BaseModel):
    provider: str = "openrouter"
    model_name: str = "deepseek/deepseek-r1"
    temperature: float = 0.7
    max_tokens: int = 4096
    provider_routing: ProviderRouting = Field(default_factory=ProviderRouting)


class ModelsConfig(BaseModel):
    heavy: ModelConfig = Field(default_factory=ModelConfig)
    light: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            model_name="deepseek/deepseek-chat",
            temperature=0.3,
            max_tokens=2048,
        )
    )


class StageQuota(BaseModel):
    perplexity_search: int = 10
    web_search: int = 5
    jina_fetch: int = 5


class QuotasConfig(BaseModel):
    broad_survey: StageQuota = Field(default_factory=StageQuota)
    deep_research: StageQuota = Field(
        default_factory=lambda: StageQuota(
            perplexity_search=20, web_search=10, jina_fetch=15
        )
    )
    writing: StageQuota = Field(
        default_factory=lambda: StageQuota(
            perplexity_search=5, web_search=3, jina_fetch=5
        )
    )


class HumanInterventionConfig(BaseModel):
    after_path_evaluation: bool = True
    after_outline_planning: bool = True
    after_deep_research: bool = False


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    quotas: QuotasConfig = Field(default_factory=QuotasConfig)
    human_intervention: HumanInterventionConfig = Field(
        default_factory=HumanInterventionConfig
    )
    stages: list[str] = Field(
        default_factory=lambda: [
            "input_preprocessing",
            "broad_survey",
            "path_evaluation",
            "deep_research",
            "writing",
            "post_processing",
        ]
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. Defaults to config.yaml in the
            backend directory.

    Returns:
        Parsed AppConfig instance.
    """
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")

    config_path = Path(config_path)
    if not config_path.exists():
        return AppConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(**raw)
