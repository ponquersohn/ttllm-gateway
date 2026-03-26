"""Main gateway orchestrator. Ties together translation, provider, and tracking.

Depends on core modules and pydantic schemas only. Database writes are handled
by the caller (API layer) using the returned metadata.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage

from ttllm.core import token_tracker, translator
from ttllm.core.provider import registry as provider_registry
from ttllm.core.streaming import format_sse_stream
from ttllm.schemas.anthropic import MessagesRequest, MessagesResponse


@dataclass
class InvokeResult:
    """Result of a non-streaming invocation."""

    response: MessagesResponse
    input_tokens: int
    output_tokens: int
    cost: Decimal
    latency_ms: int


@dataclass
class StreamResult:
    """Result metadata collected after streaming completes."""

    input_tokens: int
    output_tokens: int
    cost: Decimal
    latency_ms: int


async def invoke(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> InvokeResult:
    """Execute a non-streaming LLM request.

    Args:
        request: The parsed Anthropic-format request.
        llm_model: The ORM model with provider info and pricing.
        request_id: Correlation ID for audit.

    Returns:
        InvokeResult with the response and tracking metadata.
    """
    start = time.monotonic()

    messages = translator.to_langchain_messages(request)
    invoke_params = translator.extract_invoke_params(request)
    chat_model = provider_registry.get_chat_model(llm_model, invoke_params)

    result: AIMessage = await chat_model.ainvoke(messages)

    input_tokens, output_tokens = token_tracker.extract_token_counts(result)
    cost = token_tracker.calculate_cost(input_tokens, output_tokens, llm_model)
    latency_ms = int((time.monotonic() - start) * 1000)

    response = translator.from_langchain_response(
        result, llm_model.name, request_id, input_tokens, output_tokens
    )

    return InvokeResult(
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )


async def stream(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> tuple[AsyncIterator[str], StreamCollector]:
    """Start a streaming LLM request.

    Returns a tuple of (SSE event iterator, StreamCollector).
    The collector accumulates token counts during streaming and can be
    read after the stream is exhausted.
    """
    messages = translator.to_langchain_messages(request)
    invoke_params = translator.extract_invoke_params(request)
    chat_model = provider_registry.get_chat_model(llm_model, invoke_params)

    collector = StreamCollector(llm_model=llm_model)
    lc_stream = chat_model.astream(messages)

    sse_stream = _tracked_stream(
        format_sse_stream(lc_stream, llm_model.name, request_id),
        collector,
    )

    return sse_stream, collector


class StreamCollector:
    """Accumulates metadata during streaming."""

    def __init__(self, llm_model: Any):
        self.llm_model = llm_model
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = Decimal("0")
        self.latency_ms = 0
        self._start = time.monotonic()

    def finalize(self, input_tokens: int = 0, output_tokens: int = 0) -> StreamResult:
        self.input_tokens = input_tokens or self.input_tokens
        self.output_tokens = output_tokens or self.output_tokens
        self.cost = token_tracker.calculate_cost(
            self.input_tokens, self.output_tokens, self.llm_model
        )
        self.latency_ms = int((time.monotonic() - self._start) * 1000)
        return StreamResult(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost=self.cost,
            latency_ms=self.latency_ms,
        )


async def _tracked_stream(
    sse_stream: AsyncIterator[str],
    collector: StreamCollector,
) -> AsyncIterator[str]:
    """Wrap SSE stream to finalize the collector when done."""
    async for event in sse_stream:
        yield event
    collector.finalize()
