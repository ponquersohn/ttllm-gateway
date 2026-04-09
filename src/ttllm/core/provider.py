"""LangChain provider registry and model instantiation.

Caches model instances by model ID + config hash to avoid re-creating clients.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from ttllm.config import settings


class ProviderRegistry:
    def __init__(self, max_cache_size: int = 128):
        self._factories: dict[str, Callable] = {}
        self._cache: OrderedDict[str, BaseChatModel] = OrderedDict()
        self._max_cache_size = max_cache_size

    def clear_cache(self) -> None:
        """Evict all cached provider clients (e.g. after a secret changes)."""
        self._cache.clear()

    def register(
        self, name: str, factory: Callable[[Any, dict[str, Any]], BaseChatModel]
    ) -> None:
        self._factories[name] = factory

    @property
    def supported_providers(self) -> list[str]:
        return list(self._factories.keys())

    def get_chat_model(
        self, llm_model: Any, invoke_params: dict[str, Any]
    ) -> BaseChatModel:
        """Get or create a cached LangChain ChatModel for the given model + params."""
        cache_key = f"{llm_model.id}:{_params_hash(invoke_params)}"

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        provider = llm_model.provider
        if provider not in self._factories:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {', '.join(self._factories.keys())}"
            )

        model = self._factories[provider](llm_model, invoke_params)

        if len(self._cache) >= self._max_cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = model

        return model


def _params_hash(params: dict[str, Any]) -> str:
    """Deterministic hash for invoke params."""
    items = sorted(params.items())
    return str(hash(tuple((k, str(v)) for k, v in items)))


# --- Built-in provider factories ---


def _bedrock_factory(llm_model: Any, invoke_params: dict[str, Any]) -> BaseChatModel:
    import boto3
    from langchain_aws import ChatBedrockConverse

    config = llm_model.config_json or {}

    # Build boto3 session from explicit credentials if provided
    session_kwargs: dict[str, Any] = {}
    if config.get("aws_profile"):
        session_kwargs["profile_name"] = config["aws_profile"]
    if config.get("aws_access_key_id"):
        session_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        session_kwargs["aws_secret_access_key"] = config.get("aws_secret_access_key", "")
        if config.get("aws_session_token"):
            session_kwargs["aws_session_token"] = config["aws_session_token"]
    session_kwargs["region_name"] = config.get("region", settings.provider.default_region)

    client = boto3.Session(**session_kwargs).client("bedrock-runtime")

    return ChatBedrockConverse(
        model=llm_model.provider_model_id,
        client=client,
        max_tokens=invoke_params.get("max_tokens", 4096),
        temperature=invoke_params.get("temperature", 1.0),
    )


def _openai_factory(llm_model: Any, invoke_params: dict[str, Any]) -> BaseChatModel:
    from langchain_community.chat_models import ChatOpenAI

    config = llm_model.config_json or {}
    return ChatOpenAI(
        model=llm_model.provider_model_id,
        max_tokens=invoke_params.get("max_tokens", 4096),
        temperature=invoke_params.get("temperature", 1.0),
        openai_api_key=config.get("api_key", ""),
        openai_api_base=config.get("base_url"),
    )


# Global registry with built-in providers
registry = ProviderRegistry()
registry.register("bedrock", _bedrock_factory)
registry.register("openai", _openai_factory)
