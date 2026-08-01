"""LangChain provider + per-request state.

``LangChainProvider`` is a stateless singleton that drives OpenAI-compatible models through
the shared LangChain ``ProviderRegistry`` (``ttllm.core.providers.langchain.registry``).
``LangChainState`` accumulates one request's tokens and content, and owns the input + output
cost formula. LangChain providers do not report cache tokens, so cost has just two components.
"""

from __future__ import annotations

import time
import uuid
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.core.model import (
    ChunkKind,
    InternalChunk,
    InternalRequest,
    InternalResult,
    InternalUsage,
    Part,
    TextPart,
    ToolCallPart,
)
from ttllm.core.providers.base import BaseProvider, ProviderState
from ttllm.core.providers.langchain import translation as translator
from ttllm.core.providers.langchain.registry import registry as provider_registry


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


async def stream_chunks(lc_stream: AsyncIterator, state: Any = None) -> AsyncIterator[InternalChunk]:
    """Convert a LangChain ``astream`` into internal chunks.

    ``state`` (a duck-typed accumulator, e.g. ``LangChainState``) is an optional sink:
    when provided, the assembled text/tool content and stop reason are accumulated onto
    it, so the full response can be rebuilt after the stream ends.
    """
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    block_index = 0
    open_index: int | None = None
    open_kind: str | None = None
    # Maps a LangChain tool-call id to the block index we opened for it.
    tool_indices: dict[str, int] = {}
    text_acc = ""
    tool_acc: dict[str, dict[str, str]] = {}
    tool_order: list[str] = []

    async for chunk in lc_stream:
        text = ""
        if hasattr(chunk, "content"):
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                    elif isinstance(part, str):
                        text += part

        if text:
            if open_kind != "text":
                if open_index is not None:
                    yield InternalChunk(kind=ChunkKind.BLOCK_STOP, index=open_index)
                open_index = block_index
                block_index += 1
                open_kind = "text"
                yield InternalChunk(kind=ChunkKind.BLOCK_START, index=open_index, part=TextPart(text=""))
            text_acc += text
            yield InternalChunk(kind=ChunkKind.TEXT_DELTA, index=open_index, text=text)

        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        if tool_call_chunks:
            for tc_chunk in tool_call_chunks:
                tc_id = tc_chunk.get("id")
                tc_name = tc_chunk.get("name")
                tc_args = tc_chunk.get("args", "")

                if tc_id and tc_id not in tool_indices:
                    if open_index is not None:
                        yield InternalChunk(kind=ChunkKind.BLOCK_STOP, index=open_index)
                    idx = block_index
                    block_index += 1
                    tool_indices[tc_id] = idx
                    open_index = idx
                    open_kind = "tool"
                    tool_order.append(tc_id)
                    tool_acc[tc_id] = {"name": tc_name or "", "args": ""}
                    yield InternalChunk(
                        kind=ChunkKind.BLOCK_START, index=idx,
                        part=ToolCallPart(id=tc_id, name=tc_name or "", input={}),
                    )

                if tc_args:
                    idx = tool_indices.get(tc_id, open_index)
                    if tc_id and tc_id in tool_acc:
                        tool_acc[tc_id]["args"] += tc_args
                    if idx is not None:
                        yield InternalChunk(kind=ChunkKind.TOOL_ARGS_DELTA, index=idx, partial_json=tc_args)

        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta and isinstance(usage_meta, dict):
            input_tokens = usage_meta.get("input_tokens", input_tokens)
            output_tokens = usage_meta.get("output_tokens", output_tokens)
            input_details = usage_meta.get("input_token_details")
            if isinstance(input_details, dict):
                cache_read = input_details.get("cache_read", cache_read)

    if open_index is not None:
        yield InternalChunk(kind=ChunkKind.BLOCK_STOP, index=open_index)

    stop_reason = "tool_use" if tool_order else "end_turn"

    if state is not None:
        state.input_tokens = input_tokens
        state.output_tokens = output_tokens
        state.cache_read_tokens = cache_read
        state.stop_reason = stop_reason
        state.text = text_acc
        state.tool_calls = [
            {"id": tid, "name": tool_acc[tid]["name"], "args": tool_acc[tid]["args"]}
            for tid in tool_order
        ]

    yield InternalChunk(
        kind=ChunkKind.MESSAGE_STOP,
        stop_reason=stop_reason,
        usage=InternalUsage(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_tokens=cache_read),
    )


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
        # Non-streaming path can hand back an already-built result.
        self._response: InternalResult | None = None
        self._start = time.monotonic()
        self.latency_ms = 0
        self.error: BaseException | None = None

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

    def get_response(self) -> InternalResult:
        if self._response is not None:
            return self._response
        parts: list[Part] = []
        if self.text:
            parts.append(TextPart(text=self.text))
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
            parts.append(ToolCallPart(id=tc.get("id", ""), name=tc.get("name", ""), input=parsed))
        if not parts:
            parts.append(TextPart(text=""))
        return InternalResult(
            content=parts,
            stop_reason=self.stop_reason,
            usage=InternalUsage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class LangChainProvider(BaseProvider):
    """Stateless singleton driving OpenAI-compatible models via LangChain."""

    def _runnable(self, request: InternalRequest, llm_model: Any) -> Any:
        messages = translator.to_langchain_messages(request)
        invoke_params = translator.extract_invoke_params(request)
        chat_model = provider_registry.get_chat_model(llm_model, invoke_params)
        runnable = translator.bind_tools_to_model(chat_model, request.tools, request.tool_choice)
        return messages, runnable

    async def invoke(
        self, request: InternalRequest, llm_model: Any, request_id: uuid.UUID
    ) -> LangChainState:
        state = LangChainState(llm_model, request_id)
        messages, runnable = self._runnable(request, llm_model)

        result = await runnable.ainvoke(messages)

        input_tokens, output_tokens = _read_token_counts(result)
        state.input_tokens = input_tokens
        state.output_tokens = output_tokens
        state.raw_metadata = getattr(result, "response_metadata", {}) or {}
        internal_result = translator.from_langchain_response(result, input_tokens, output_tokens)
        state.stop_reason = internal_result.stop_reason
        state._response = internal_result
        state.mark_finished()
        return state

    def stream(
        self, request: InternalRequest, llm_model: Any, request_id: uuid.UUID
    ) -> tuple[LangChainState, AsyncIterator[InternalChunk]]:
        state = LangChainState(llm_model, request_id)

        async def _gen() -> AsyncIterator[InternalChunk]:
            try:
                messages, runnable = self._runnable(request, llm_model)
                lc_stream = runnable.astream(messages)
                async for chunk in stream_chunks(lc_stream, state=state):
                    yield chunk
            finally:
                state.mark_finished()

        return state, _gen()
