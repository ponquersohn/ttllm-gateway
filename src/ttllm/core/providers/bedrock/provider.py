"""Bedrock provider + per-request state.

``BedrockProvider`` is a stateless singleton that drives the boto3 Converse API (the heavy
client/executor machinery lives in ``ttllm.core.providers.bedrock.converse``). ``BedrockState``
accumulates one request's tokens, cache counts, raw usage payload and assembled content, and
owns the Bedrock cost formula (input + output + cache read + cache write) and metadata blob.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.core.model import InternalChunk, InternalRequest, InternalResult, InternalUsage, Part, TextPart
from ttllm.core.providers.base import BaseProvider, ProviderState
from ttllm.core.providers.bedrock import converse as bedrock


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
        self.content_parts: list[Part] = []
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

    def get_response(self) -> InternalResult:
        parts = self.content_parts or [TextPart(text="")]
        return InternalResult(
            content=parts,
            stop_reason=self.stop_reason,
            usage=InternalUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                cache_read_tokens=self.cache_read_tokens,
                cache_write_tokens=self.cache_write_tokens,
            ),
        )


class BedrockProvider(BaseProvider):
    """Stateless singleton driving the Bedrock Converse API."""

    async def invoke(
        self, request: InternalRequest, llm_model: Any, request_id: uuid.UUID
    ) -> BedrockState:
        state = BedrockState(llm_model, request_id)
        raw = await bedrock._converse_raw(request, llm_model)
        result = bedrock.parse_converse_response(raw)
        state.input_tokens = result.usage.input_tokens
        state.output_tokens = result.usage.output_tokens
        state.cache_read_tokens = result.usage.cache_read_tokens
        state.cache_write_tokens = result.usage.cache_write_tokens
        state.raw_usage = raw.get("usage", {})
        state.stop_reason = result.stop_reason
        state.content_parts = list(result.content)
        state.mark_finished()
        return state

    def stream(
        self, request: InternalRequest, llm_model: Any, request_id: uuid.UUID
    ) -> tuple[BedrockState, AsyncIterator[InternalChunk]]:
        state = BedrockState(llm_model, request_id)

        async def _gen() -> AsyncIterator[InternalChunk]:
            try:
                async for chunk in bedrock.stream_converse_chunks(
                    request, llm_model, request_id, state=state
                ):
                    yield chunk
            finally:
                state.mark_finished()

        return state, _gen()
