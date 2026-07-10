"""Bedrock provider + per-request state.

``BedrockProvider`` is a stateless singleton that drives the boto3 Converse API (the heavy
client/executor machinery lives in ``ttllm.core.bedrock``). ``BedrockState`` accumulates one
request's tokens, cache counts, raw usage payload and assembled content, and owns the Bedrock
cost formula (input + output + cache read + cache write) and metadata blob.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.core import bedrock
from ttllm.core.providers.base import BaseProvider, ProviderState
from ttllm.schemas.anthropic import (
    ContentBlock,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    Usage,
)


def _cost(tokens: int, rate: Any) -> Decimal:
    return (Decimal(tokens) / 1000) * Decimal(str(rate or 0))


class BedrockState(ProviderState):
    """Per-request accumulator for a Bedrock Converse exchange."""

    def __init__(self, llm_model: Any, request_id: uuid.UUID):
        self.llm_model = llm_model
        self.request_id = request_id
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.raw_usage: dict[str, Any] = {}
        self.stop_reason = "end_turn"
        self.content_blocks: list[ContentBlock] = []
        self._start = time.monotonic()
        self.latency_ms = 0
        self.error: BaseException | None = None

    def mark_finished(self) -> None:
        """Stamp elapsed time. Called by the provider once the exchange completes."""
        self.latency_ms = int((time.monotonic() - self._start) * 1000)

    def _cost_components(self) -> dict[str, Decimal]:
        m = self.llm_model
        return {
            "input": _cost(self.input_tokens, m.input_cost_per_1k),
            "output": _cost(self.output_tokens, m.output_cost_per_1k),
            "cache_read": _cost(self.cache_read_tokens, getattr(m, "cache_read_cost_per_1k", 0)),
            "cache_write": _cost(self.cache_write_tokens, getattr(m, "cache_write_cost_per_1k", 0)),
        }

    def get_cost(self) -> Decimal:
        return sum(self._cost_components().values(), Decimal("0"))

    def get_metadata(self) -> dict[str, Any]:
        components = self._cost_components()
        return {
            "provider": "bedrock",
            "raw": self.raw_usage,
            "stop_reason": self.stop_reason,
            "latency_ms": self.latency_ms,
            "cost": {
                "total": str(self.get_cost()),
                "components": {k: str(v) for k, v in components.items()},
                "tokens": {
                    "input": self.input_tokens,
                    "output": self.output_tokens,
                    "cache_read": self.cache_read_tokens,
                    "cache_write": self.cache_write_tokens,
                },
            },
        }

    def get_response(self) -> MessagesResponse:
        blocks = self.content_blocks or [TextBlock(text="")]
        return MessagesResponse(
            id=f"msg_{self.request_id.hex[:24]}",
            content=blocks,
            model=self.llm_model.name,
            stop_reason=self.stop_reason,
            usage=Usage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_creation_input_tokens=self.cache_write_tokens or None,
                cache_read_input_tokens=self.cache_read_tokens or None,
            ),
        )


class BedrockProvider(BaseProvider):
    """Stateless singleton driving the Bedrock Converse API."""

    async def invoke(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> BedrockState:
        state = BedrockState(llm_model, request_id)
        raw = await bedrock._converse_raw(request, llm_model)
        response, cache_read, cache_write = bedrock.parse_converse_response(
            raw, llm_model.name, request_id
        )
        state.input_tokens = response.usage.input_tokens
        state.output_tokens = response.usage.output_tokens
        state.cache_read_tokens = cache_read
        state.cache_write_tokens = cache_write
        state.raw_usage = raw.get("usage", {})
        state.stop_reason = response.stop_reason or "end_turn"
        state.content_blocks = list(response.content)
        state.mark_finished()
        return state

    def stream(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> tuple[BedrockState, AsyncIterator[str]]:
        state = BedrockState(llm_model, request_id)

        async def _gen() -> AsyncIterator[str]:
            try:
                async for event in bedrock.stream_converse(
                    request, llm_model, request_id, state=state
                ):
                    yield event
            finally:
                state.mark_finished()

        return state, _gen()
