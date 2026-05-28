"""Main gateway orchestrator. Ties together translation, provider, and tracking.

Routes Bedrock requests directly through boto3 Converse API (no LangChain).
Routes OpenAI-compatible requests through LangChain (unchanged).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.core import token_tracker
from ttllm.schemas.anthropic import MessagesRequest, MessagesResponse, ServerToolDefinition


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


class ServerToolError(Exception):
    """Raised when a request contains server-side tools that cannot be proxied."""

    pass


def _has_server_tools(request: MessagesRequest) -> bool:
    if not request.tools:
        return False
    return any(isinstance(t, ServerToolDefinition) for t in request.tools)


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
    if _has_server_tools(request):
        raise ServerToolError(
            "Server-side tools (web_search, code_execution) cannot be proxied through the gateway. "
            "Remove server tool definitions and handle them client-side."
        )

    start = time.monotonic()

    if llm_model.provider == "bedrock":
        response = await _invoke_bedrock(request, llm_model, request_id)
    else:
        response = await _invoke_langchain(request, llm_model, request_id)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = token_tracker.calculate_cost(input_tokens, output_tokens, llm_model)
    latency_ms = int((time.monotonic() - start) * 1000)

    return InvokeResult(
        response=response,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )


async def _invoke_bedrock(
    request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
) -> MessagesResponse:
    from ttllm.core.bedrock import invoke_converse

    return await invoke_converse(request, llm_model, request_id)


async def _invoke_langchain(
    request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
) -> MessagesResponse:
    from langchain_core.messages import AIMessage

    from ttllm.core import translator
    from ttllm.core.provider import registry as provider_registry

    messages = translator.to_langchain_messages(request)
    invoke_params = translator.extract_invoke_params(request)
    chat_model = provider_registry.get_chat_model(llm_model, invoke_params)
    runnable = translator.bind_tools_to_model(chat_model, request.tools, request.tool_choice)

    result: AIMessage = await runnable.ainvoke(messages)

    input_tokens, output_tokens = token_tracker.extract_token_counts(result)

    return translator.from_langchain_response(
        result, llm_model.name, request_id, input_tokens, output_tokens
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
    if _has_server_tools(request):
        raise ServerToolError(
            "Server-side tools (web_search, code_execution) cannot be proxied through the gateway. "
            "Remove server tool definitions and handle them client-side."
        )

    collector = StreamCollector(llm_model=llm_model)

    if llm_model.provider == "bedrock":
        sse_stream = _stream_bedrock(request, llm_model, request_id, collector)
    else:
        sse_stream = _stream_langchain(request, llm_model, request_id, collector)

    return sse_stream, collector


async def _stream_bedrock(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
    collector: StreamCollector,
) -> AsyncIterator[str]:
    from ttllm.core.bedrock import stream_converse

    async for event in stream_converse(request, llm_model, request_id):
        yield event

    collector.finalize()


async def _stream_langchain(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
    collector: StreamCollector,
) -> AsyncIterator[str]:
    from ttllm.core import translator
    from ttllm.core.provider import registry as provider_registry
    from ttllm.core.streaming import format_sse_stream

    messages = translator.to_langchain_messages(request)
    invoke_params = translator.extract_invoke_params(request)
    chat_model = provider_registry.get_chat_model(llm_model, invoke_params)
    runnable = translator.bind_tools_to_model(chat_model, request.tools, request.tool_choice)

    lc_stream = runnable.astream(messages)
    token_usage: dict[str, int] = {}

    async for event in format_sse_stream(lc_stream, llm_model.name, request_id, token_usage):
        yield event

    collector.input_tokens = token_usage.get("input_tokens", 0)
    collector.output_tokens = token_usage.get("output_tokens", 0)
    collector.finalize()


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
