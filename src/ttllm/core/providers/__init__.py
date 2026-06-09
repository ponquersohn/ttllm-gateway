"""Provider registry: select a singleton provider from an LLM model.

Providers are long-lived stateless singletons (they own connection caches / registries).
``get_provider`` maps ``llm_model.provider`` to one; anything not explicitly registered
falls through to LangChain — mirroring the gateway's historical "bedrock else LangChain"
dispatch.
"""

from __future__ import annotations

from typing import Any

from ttllm.core.providers.base import BaseProvider, ProviderState
from ttllm.core.providers.bedrock_provider import BedrockProvider
from ttllm.core.providers.langchain_provider import LangChainProvider

_BEDROCK = BedrockProvider()
_LANGCHAIN = LangChainProvider()

_PROVIDERS: dict[str, BaseProvider] = {
    "bedrock": _BEDROCK,
}


def get_provider(llm_model: Any) -> BaseProvider:
    """Return the singleton provider for the given model's ``provider`` field."""
    return _PROVIDERS.get(llm_model.provider, _LANGCHAIN)


__all__ = ["BaseProvider", "ProviderState", "get_provider"]
