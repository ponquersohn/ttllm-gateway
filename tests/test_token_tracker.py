"""Tests for token counting and cost calculation."""

from decimal import Decimal
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from ttllm.core.token_tracker import calculate_cost, extract_token_counts


class TestCalculateCost:
    def test_basic_cost(self):
        model = MagicMock()
        model.input_cost_per_1k = Decimal("0.003")
        model.output_cost_per_1k = Decimal("0.015")

        cost = calculate_cost(1000, 500, model)
        expected = Decimal("0.003") + (Decimal("500") / 1000) * Decimal("0.015")
        assert cost == expected

    def test_zero_tokens(self):
        model = MagicMock()
        model.input_cost_per_1k = Decimal("0.003")
        model.output_cost_per_1k = Decimal("0.015")

        cost = calculate_cost(0, 0, model)
        assert cost == Decimal("0")


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
