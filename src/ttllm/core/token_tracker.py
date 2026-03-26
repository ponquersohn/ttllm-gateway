"""Token counting and cost calculation. No framework dependencies."""

from decimal import Decimal
from typing import Any


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    llm_model: Any,
) -> Decimal:
    """Calculate the cost based on token counts and model pricing.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        llm_model: Model object with input_cost_per_1k and output_cost_per_1k.

    Returns:
        Total cost as a Decimal.
    """
    input_cost = (Decimal(input_tokens) / 1000) * Decimal(
        str(llm_model.input_cost_per_1k)
    )
    output_cost = (Decimal(output_tokens) / 1000) * Decimal(
        str(llm_model.output_cost_per_1k)
    )
    return input_cost + output_cost


def extract_token_counts(response: Any) -> tuple[int, int]:
    """Extract input and output token counts from a LangChain response.

    Returns:
        Tuple of (input_tokens, output_tokens).
    """
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        return (
            usage_meta.get("input_tokens", 0),
            usage_meta.get("output_tokens", 0),
        )
    # Fallback: check response_metadata
    resp_meta = getattr(response, "response_metadata", {})
    usage = resp_meta.get("usage", {})
    if usage:
        return (
            usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            usage.get("completion_tokens", usage.get("output_tokens", 0)),
        )
    return (0, 0)
