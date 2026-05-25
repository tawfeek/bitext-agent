"""Factory for the Nebius Token Factory chat model.

Nebius Token Factory exposes an OpenAI-compatible endpoint, so we use
``ChatOpenAI`` pointed at Nebius' base URL.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import SETTINGS


def build_chat_model(temperature: float = 0.0, **kwargs) -> ChatOpenAI:
    """Return a ChatOpenAI configured for Nebius Token Factory."""
    if not SETTINGS.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=SETTINGS.nebius_model,
        api_key=SETTINGS.nebius_api_key,
        base_url=SETTINGS.nebius_base_url,
        temperature=temperature,
        **kwargs,
    )
