"""Tests for token counting and cost calculation."""

from decimal import Decimal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from ttllm.core.token_tracker import calculate_cost, extract_token_counts


class _Pricing:
    """Minimal model stand-in with explicit pricing attributes."""

    def __init__(self, **kwargs):
        self.input_cost_per_1k = Decimal("0")
        self.output_cost_per_1k = Decimal("0")
        self.cache_read_cost_per_1k = Decimal("0")
        self.cache_write_cost_per_1k = Decimal("0")
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCalculateCost:
    def test_basic_cost(self):
        model = _Pricing(input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015"))

        cost = calculate_cost(1000, 500, model)
        expected = Decimal("0.003") + (Decimal("500") / 1000) * Decimal("0.015")
        assert cost == expected

    def test_zero_tokens(self):
        model = _Pricing(input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015"))

        cost = calculate_cost(0, 0, model)
        assert cost == Decimal("0")

    def test_cache_cost(self):
        model = _Pricing(
            input_cost_per_1k=Decimal("0.003"),
            output_cost_per_1k=Decimal("0.015"),
            cache_read_cost_per_1k=Decimal("0.0003"),
            cache_write_cost_per_1k=Decimal("0.00375"),
        )

        cost = calculate_cost(1000, 500, model, cache_read_tokens=2000, cache_write_tokens=1000)
        expected = (
            (Decimal("1000") / 1000) * Decimal("0.003")
            + (Decimal("500") / 1000) * Decimal("0.015")
            + (Decimal("2000") / 1000) * Decimal("0.0003")
            + (Decimal("1000") / 1000) * Decimal("0.00375")
        )
        assert cost == expected

    def test_backward_compat_model_without_cache_fields(self):
        """Models lacking cache pricing attributes still compute input+output cost."""

        class _Legacy:
            input_cost_per_1k = Decimal("0.003")
            output_cost_per_1k = Decimal("0.015")

        model = _Legacy()
        cost = calculate_cost(1000, 500, model, cache_read_tokens=2000, cache_write_tokens=1000)
        expected = Decimal("0.003") + (Decimal("500") / 1000) * Decimal("0.015")
        assert cost == expected


class TestExtractTokenCounts:
    def test_from_usage_metadata(self):
        msg = AIMessage(content="test")
        msg.usage_metadata = {"input_tokens": 42, "output_tokens": 17}
        input_t, output_t = extract_token_counts(msg)
        assert input_t == 42
        assert output_t == 17

    def test_from_response_metadata(self):
        msg = AIMessage(content="test")
        msg.response_metadata = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20}
        }
        input_t, output_t = extract_token_counts(msg)
        assert input_t == 10
        assert output_t == 20

    def test_no_metadata(self):
        msg = AIMessage(content="test")
        input_t, output_t = extract_token_counts(msg)
        assert input_t == 0
        assert output_t == 0
