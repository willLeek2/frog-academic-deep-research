"""LLM instance factory.

Creates *heavy* and *light* LLM instances based on configuration. Tries the
official ``langchain-openrouter`` package first (Scheme A), and falls back to
``langchain-openai`` with a custom ``base_url`` (Scheme B) if the former is
not available or raises an error.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel

from core.config_loader import AppConfig, ModelConfig


def _build_provider_kwargs(cfg: ModelConfig) -> dict[str, Any]:
    """Return the provider routing dict for OpenRouter."""
    return {
        "order": cfg.provider_routing.order,
        "allow_fallbacks": cfg.provider_routing.allow_fallbacks,
        "require_parameters": cfg.provider_routing.require_parameters,
    }


def _create_llm(cfg: ModelConfig) -> BaseChatModel:
    """Create a single LLM instance with automatic fallback."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # Scheme A: try langchain-openrouter
    try:
        from langchain_openrouter import ChatOpenRouter  # type: ignore[import-untyped]

        return ChatOpenRouter(
            model_name=cfg.model_name,
            openrouter_api_key=api_key,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            openrouter_provider=_build_provider_kwargs(cfg),
        )
    except Exception:
        pass

    # Scheme B: langchain-openai with custom base_url
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.model_name,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,  # type: ignore[arg-type]
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        model_kwargs={"provider": _build_provider_kwargs(cfg)},
    )


def create_heavy_llm(config: AppConfig) -> BaseChatModel:
    """Create the *heavy* LLM instance (for complex reasoning tasks)."""
    return _create_llm(config.models.heavy)


def create_light_llm(config: AppConfig) -> BaseChatModel:
    """Create the *light* LLM instance (for simpler extraction tasks)."""
    return _create_llm(config.models.light)
