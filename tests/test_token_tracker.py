"""Tests for provider-owned cost calculation and token reading.

Cost math now lives on each provider's per-request state (``BedrockState`` /
``LangChainState``), and LangChain token reading is folded into ``LangChainState``.
"""

import uuid
from decimal import Decimal

from langchain_core.messages import AIMessage

from ttllm.core.providers.bedrock.provider import BedrockState
from ttllm.core.providers.langchain.provider import LangChainState, _read_token_counts


class _Pricing:
    """Minimal model stand-in with explicit pricing attributes."""

    def __init__(self, **kwargs):
        self.name = "test-model"
        self.input_cost_per_1k = Decimal("0")
        self.output_cost_per_1k = Decimal("0")
        self.cache_read_cost_per_1k = Decimal("0")
        self.cache_write_cost_per_1k = Decimal("0")
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestBedrockStateCost:
    def test_basic_cost(self):
        model = _Pricing(input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015"))
        state = BedrockState(model, uuid.uuid4())
        state.input_tokens = 1000
        state.output_tokens = 500

        expected = Decimal("0.003") + (Decimal("500") / 1000) * Decimal("0.015")
        assert state.get_cost() == expected

    def test_zero_tokens(self):
        model = _Pricing(input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015"))
        state = BedrockState(model, uuid.uuid4())
        assert state.get_cost() == Decimal("0")

    def test_cache_cost(self):
        model = _Pricing(
            input_cost_per_1k=Decimal("0.003"),
            output_cost_per_1k=Decimal("0.015"),
            cache_read_cost_per_1k=Decimal("0.0003"),
            cache_write_cost_per_1k=Decimal("0.00375"),
        )
        state = BedrockState(model, uuid.uuid4())
        state.input_tokens = 1000
        state.output_tokens = 500
        state.cache_read_tokens = 2000
        state.cache_write_tokens = 1000

        expected = (
            (Decimal("1000") / 1000) * Decimal("0.003")
            + (Decimal("500") / 1000) * Decimal("0.015")
            + (Decimal("2000") / 1000) * Decimal("0.0003")
            + (Decimal("1000") / 1000) * Decimal("0.00375")
        )
        assert state.get_cost() == expected

    def test_metadata_blob_has_breakdown(self):
        model = _Pricing(
            input_cost_per_1k=Decimal("0.003"),
            output_cost_per_1k=Decimal("0.015"),
            cache_read_cost_per_1k=Decimal("0.0003"),
            cache_write_cost_per_1k=Decimal("0.00375"),
        )
        state = BedrockState(model, uuid.uuid4())
        state.input_tokens = 1000
        state.output_tokens = 500
        state.cache_read_tokens = 2000
        state.cache_write_tokens = 1000
        state.raw_usage = {"inputTokens": 1000}

        meta = state.get_metadata()
        assert meta["provider"] == "bedrock"
        assert meta["raw"] == {"inputTokens": 1000}
        assert set(meta["cost"]["components"]) == {"input", "output", "cache_read", "cache_write", "cache_write_1h"}
        assert meta["cost"]["total"] == str(state.get_cost())
        assert meta["cost"]["tokens"]["cache_read"] == 2000


class TestLangChainStateCost:
    def test_input_output_only(self):
        model = _Pricing(input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015"))
        state = LangChainState(model, uuid.uuid4())
        state.input_tokens = 1000
        state.output_tokens = 500

        expected = Decimal("0.003") + (Decimal("500") / 1000) * Decimal("0.015")
        assert state.get_cost() == expected

        meta = state.get_metadata()
        assert meta["provider"] == "langchain"
        assert set(meta["cost"]["components"]) == {"input", "output"}


class TestReadTokenCounts:
    def test_from_usage_metadata(self):
        msg = AIMessage(content="test")
        msg.usage_metadata = {"input_tokens": 42, "output_tokens": 17}
        assert _read_token_counts(msg) == (42, 17)

    def test_from_response_metadata(self):
        msg = AIMessage(content="test")
        msg.response_metadata = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        assert _read_token_counts(msg) == (10, 20)

    def test_no_metadata(self):
        msg = AIMessage(content="test")
        assert _read_token_counts(msg) == (0, 0)


class TestBedrockCacheWrite1hCost:
    def test_1h_writes_billed_at_twice_input_rate(self):
        model = _Pricing(
            input_cost_per_1k=Decimal("0.004"),
            output_cost_per_1k=Decimal("0.02"),
            cache_read_cost_per_1k=Decimal("0.0004"),
            cache_write_cost_per_1k=Decimal("0.005"),
        )
        state = BedrockState(model, uuid.uuid4())
        state.cache_write_tokens = 3000
        state.cache_write_1h_tokens = 1000

        components = state.get_metadata()["cost"]["components"]
        assert Decimal(components["cache_write"]) == (Decimal("2000") / 1000) * Decimal("0.005")
        assert Decimal(components["cache_write_1h"]) == (Decimal("1000") / 1000) * Decimal("0.008")
        assert state.get_cost() == Decimal("0.010") + Decimal("0.008")
        assert state.get_response().usage.cache_write_1h_tokens == 1000
