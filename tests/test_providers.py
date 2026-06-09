"""Tests for the provider abstraction: selection, Bedrock invoke/stream, state population."""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from ttllm.core.providers import get_provider
from ttllm.core.providers.bedrock_provider import BedrockProvider, BedrockState
from ttllm.core.providers.langchain_provider import LangChainProvider
from ttllm.schemas.anthropic import Message, MessagesRequest


def _make_model(provider="bedrock", **overrides):
    model = MagicMock()
    model.provider = provider
    model.provider_model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
    model.name = "claude-sonnet"
    model.config_json = {"region": "us-east-1"}
    model.input_cost_per_1k = Decimal("0.003")
    model.output_cost_per_1k = Decimal("0.015")
    model.cache_read_cost_per_1k = Decimal("0.0003")
    model.cache_write_cost_per_1k = Decimal("0.00375")
    for k, v in overrides.items():
        setattr(model, k, v)
    return model


def _make_request(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-sonnet",
        "max_tokens": 1024,
        "messages": [Message(role="user", content="Hello")],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


def _make_stream_response(events):
    class RecordingStream:
        def __init__(self, items):
            self._items = list(items)

        def __iter__(self):
            return self

        def __next__(self):
            if not self._items:
                raise StopIteration
            return self._items.pop(0)

    return {"stream": RecordingStream(events)}


class TestGetProvider:
    def test_bedrock(self):
        assert isinstance(get_provider(_make_model(provider="bedrock")), BedrockProvider)

    def test_openai_falls_to_langchain(self):
        assert isinstance(get_provider(_make_model(provider="openai")), LangChainProvider)

    def test_unknown_falls_to_langchain(self):
        assert isinstance(get_provider(_make_model(provider="something-new")), LangChainProvider)

    def test_singletons_are_reused(self):
        a = get_provider(_make_model(provider="bedrock"))
        b = get_provider(_make_model(provider="bedrock"))
        assert a is b


class TestBedrockProviderInvoke:
    @pytest.mark.asyncio
    async def test_invoke_populates_state(self):
        provider = BedrockProvider()
        model = _make_model()

        raw_response = {
            "output": {"message": {"content": [{"text": "Hello there"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 40,
                "cacheReadInputTokens": 50,
                "cacheWriteInputTokens": 30,
            },
        }

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse.return_value = raw_response
            mock_get_client.return_value = mock_client

            state = await provider.invoke(_make_request(), model, uuid.uuid4())

        assert isinstance(state, BedrockState)
        assert state.input_tokens == 100
        assert state.output_tokens == 40
        assert state.cache_read_tokens == 50
        assert state.cache_write_tokens == 30

        expected = (
            (Decimal("100") / 1000) * Decimal("0.003")
            + (Decimal("40") / 1000) * Decimal("0.015")
            + (Decimal("50") / 1000) * Decimal("0.0003")
            + (Decimal("30") / 1000) * Decimal("0.00375")
        )
        assert state.get_cost() == expected
        assert state.get_response().content[0].text == "Hello there"
        meta = state.get_metadata()
        assert meta["provider"] == "bedrock"
        assert meta["raw"]["inputTokens"] == 100


class TestBedrockProviderStream:
    @pytest.mark.asyncio
    async def test_stream_populates_state(self):
        provider = BedrockProvider()
        model = _make_model()

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi "}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "there"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {
                "inputTokens": 10, "outputTokens": 5,
                "cacheReadInputTokens": 0, "cacheWriteInputTokens": 0,
            }}},
        ]

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            state, sse = provider.stream(_make_request(), model, uuid.uuid4())
            collected = [ev async for ev in sse]

        # SSE was emitted to the client AND the state was populated for finalize.
        assert any("message_stop" in ev for ev in collected)
        assert state.input_tokens == 10
        assert state.output_tokens == 5
        # Rebuilt response reassembles the text deltas.
        assert state.get_response().content[0].text == "Hi there"
        assert state.latency_ms >= 0
