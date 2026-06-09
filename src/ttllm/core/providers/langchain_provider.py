"""LangChain provider + per-request state.

``LangChainProvider`` is a stateless singleton that drives OpenAI-compatible models through
the shared LangChain ``ProviderRegistry`` (``ttllm.core.provider``). ``LangChainState``
accumulates one request's tokens and content, and owns the input + output cost formula.
LangChain providers do not report cache tokens, so cost has just two components.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.core import translator
from ttllm.core.provider import registry as provider_registry
from ttllm.core.providers.base import BaseProvider, ProviderState
from ttllm.core.streaming import format_sse_stream
from ttllm.schemas.anthropic import (
    ContentBlock,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)


def _cost(tokens: int, rate: Any) -> Decimal:
    return (Decimal(tokens) / 1000) * Decimal(str(rate or 0))


def _read_token_counts(response: Any) -> tuple[int, int]:
    """Read (input, output) token counts from a LangChain AIMessage.

    Tries ``usage_metadata`` first, then falls back to ``response_metadata['usage']``.
    (Folded in from the former ``token_tracker.extract_token_counts``.)
    """
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        return (
            usage_meta.get("input_tokens", 0),
            usage_meta.get("output_tokens", 0),
        )
    resp_meta = getattr(response, "response_metadata", {}) or {}
    usage = resp_meta.get("usage", {})
    if usage:
        return (
            usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            usage.get("completion_tokens", usage.get("output_tokens", 0)),
        )
    return (0, 0)


class LangChainState(ProviderState):
    """Per-request accumulator for a LangChain exchange."""

    def __init__(self, llm_model: Any, request_id: uuid.UUID):
        self.llm_model = llm_model
        self.request_id = request_id
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.stop_reason = "end_turn"
        self.text = ""
        # list of {"id", "name", "args"(json str)} for streaming, or rich dicts for invoke
        self.tool_calls: list[dict[str, Any]] = []
        self.raw_metadata: dict[str, Any] = {}
        # Non-streaming path can hand back an already-built response.
        self._response: MessagesResponse | None = None
        self._start = time.monotonic()
        self.latency_ms = 0

    def mark_finished(self) -> None:
        self.latency_ms = int((time.monotonic() - self._start) * 1000)

    def _cost_components(self) -> dict[str, Decimal]:
        m = self.llm_model
        return {
            "input": _cost(self.input_tokens, m.input_cost_per_1k),
            "output": _cost(self.output_tokens, m.output_cost_per_1k),
        }

    def get_cost(self) -> Decimal:
        return sum(self._cost_components().values(), Decimal("0"))

    def get_metadata(self) -> dict[str, Any]:
        components = self._cost_components()
        return {
            "provider": "langchain",
            "raw": self.raw_metadata,
            "stop_reason": self.stop_reason,
            "latency_ms": self.latency_ms,
            "cost": {
                "total": str(self.get_cost()),
                "components": {k: str(v) for k, v in components.items()},
                "tokens": {"input": self.input_tokens, "output": self.output_tokens},
            },
        }

    def get_response(self) -> MessagesResponse:
        if self._response is not None:
            return self._response
        blocks: list[ContentBlock] = []
        if self.text:
            blocks.append(TextBlock(text=self.text))
        for tc in self.tool_calls:
            args = tc.get("args", "")
            if isinstance(args, str):
                import json

                try:
                    parsed = json.loads(args) if args else {}
                except (ValueError, TypeError):
                    parsed = {}
            else:
                parsed = args
            blocks.append(ToolUseBlock(id=tc.get("id", ""), name=tc.get("name", ""), input=parsed))
        if not blocks:
            blocks.append(TextBlock(text=""))
        return MessagesResponse(
            id=f"msg_{self.request_id.hex[:24]}",
            content=blocks,
            model=self.llm_model.name,
            stop_reason=self.stop_reason,
            usage=Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class LangChainProvider(BaseProvider):
    """Stateless singleton driving OpenAI-compatible models via LangChain."""

    def _runnable(self, request: MessagesRequest, llm_model: Any) -> Any:
        messages = translator.to_langchain_messages(request)
        invoke_params = translator.extract_invoke_params(request)
        chat_model = provider_registry.get_chat_model(llm_model, invoke_params)
        runnable = translator.bind_tools_to_model(chat_model, request.tools, request.tool_choice)
        return messages, runnable

    async def invoke(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> LangChainState:
        state = LangChainState(llm_model, request_id)
        messages, runnable = self._runnable(request, llm_model)

        result = await runnable.ainvoke(messages)

        input_tokens, output_tokens = _read_token_counts(result)
        state.input_tokens = input_tokens
        state.output_tokens = output_tokens
        state.raw_metadata = getattr(result, "response_metadata", {}) or {}
        response = translator.from_langchain_response(
            result, llm_model.name, request_id, input_tokens, output_tokens
        )
        state.stop_reason = response.stop_reason or "end_turn"
        state._response = response
        state.mark_finished()
        return state

    def stream(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> tuple[LangChainState, AsyncIterator[str]]:
        state = LangChainState(llm_model, request_id)

        async def _gen() -> AsyncIterator[str]:
            try:
                messages, runnable = self._runnable(request, llm_model)
                lc_stream = runnable.astream(messages)
                async for event in format_sse_stream(
                    lc_stream, llm_model.name, request_id, state=state
                ):
                    yield event
            finally:
                state.mark_finished()

        return state, _gen()
