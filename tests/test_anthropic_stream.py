"""Tests for the shared internal-chunk -> Anthropic SSE encoder."""

from __future__ import annotations

import json
import uuid

import pytest

from ttllm.core.adapters.anthropic_stream import encode_anthropic_sse
from ttllm.core.model import ChunkKind, InternalChunk, InternalUsage, TextPart, ThinkingPart, ToolCallPart


async def _chunks(items):
    for item in items:
        yield item


def _parse_sse(raw: str):
    lines = raw.strip().split("\n")
    event_type = lines[0].replace("event: ", "")
    data = json.loads(lines[1].replace("data: ", ""))
    return event_type, data


async def _collect(items):
    return [ev async for ev in encode_anthropic_sse(_chunks(items), "claude-sonnet", uuid.uuid4())]


class TestEncodeAnthropicSse:
    @pytest.mark.asyncio
    async def test_message_start_and_ping_always_first(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="end_turn", usage=InternalUsage(input_tokens=0, output_tokens=0)),
        ])
        types = [_parse_sse(e)[0] for e in collected]
        assert types[:2] == ["message_start", "ping"]

    @pytest.mark.asyncio
    async def test_text_stream(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=TextPart(text="")),
            InternalChunk(kind=ChunkKind.TEXT_DELTA, index=0, text="Hello"),
            InternalChunk(kind=ChunkKind.TEXT_DELTA, index=0, text=" world"),
            InternalChunk(kind=ChunkKind.BLOCK_STOP, index=0),
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="end_turn", usage=InternalUsage(input_tokens=5, output_tokens=2)),
        ])
        types = [_parse_sse(e)[0] for e in collected]
        assert types == [
            "message_start", "ping", "content_block_start", "content_block_delta",
            "content_block_delta", "content_block_stop", "message_delta", "message_stop",
        ]
        deltas = [_parse_sse(e)[1]["delta"]["text"] for e in collected if e.startswith("event: content_block_delta")]
        assert "".join(deltas) == "Hello world"
        message_delta = _parse_sse(next(e for e in collected if e.startswith("event: message_delta")))[1]
        assert message_delta["delta"]["stop_reason"] == "end_turn"
        assert message_delta["usage"]["input_tokens"] == 5
        assert message_delta["usage"]["output_tokens"] == 2

    @pytest.mark.asyncio
    async def test_tool_use_stream(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=ToolCallPart(id="tu_1", name="search", input={})),
            InternalChunk(kind=ChunkKind.TOOL_ARGS_DELTA, index=0, partial_json='{"q":'),
            InternalChunk(kind=ChunkKind.TOOL_ARGS_DELTA, index=0, partial_json='"x"}'),
            InternalChunk(kind=ChunkKind.BLOCK_STOP, index=0),
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="tool_use", usage=InternalUsage(input_tokens=5, output_tokens=2)),
        ])
        start = _parse_sse(next(e for e in collected if e.startswith("event: content_block_start")))[1]
        assert start["content_block"]["type"] == "tool_use"
        assert start["content_block"]["id"] == "tu_1"
        partials = [_parse_sse(e)[1]["delta"]["partial_json"] for e in collected if e.startswith("event: content_block_delta")]
        assert "".join(partials) == '{"q":"x"}'

    @pytest.mark.asyncio
    async def test_thinking_stream(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=ThinkingPart(text="", signature="")),
            InternalChunk(kind=ChunkKind.THINKING_DELTA, index=0, text="pondering"),
            InternalChunk(kind=ChunkKind.SIGNATURE_DELTA, index=0, signature="sig_abc"),
            InternalChunk(kind=ChunkKind.BLOCK_STOP, index=0),
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="end_turn", usage=InternalUsage(input_tokens=1, output_tokens=1)),
        ])
        start = _parse_sse(next(e for e in collected if e.startswith("event: content_block_start")))[1]
        assert start["content_block"]["type"] == "thinking"
        deltas = [_parse_sse(e)[1]["delta"] for e in collected if e.startswith("event: content_block_delta")]
        assert deltas[0] == {"type": "thinking_delta", "thinking": "pondering"}
        assert deltas[1] == {"type": "signature_delta", "signature": "sig_abc"}

    @pytest.mark.asyncio
    async def test_error_chunk_stops_stream(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=TextPart(text="")),
            InternalChunk(kind=ChunkKind.ERROR, error_message="boom"),
            # never reached -- the encoder returns immediately on ERROR.
            InternalChunk(kind=ChunkKind.TEXT_DELTA, index=0, text="unreachable"),
        ])
        types = [_parse_sse(e)[0] for e in collected]
        assert types[-1] == "error"
        error_data = _parse_sse(collected[-1])[1]
        assert error_data["error"]["message"] == "boom"
        assert "message_delta" not in types
        assert not any("unreachable" in e for e in collected)

    @pytest.mark.asyncio
    async def test_unclosed_block_closed_defensively(self):
        # No BLOCK_STOP for index 0 -- the encoder must still close it before
        # message_delta, or a real Anthropic client would reject the stream.
        collected = await _collect([
            InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=TextPart(text="")),
            InternalChunk(kind=ChunkKind.TEXT_DELTA, index=0, text="hi"),
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="end_turn", usage=InternalUsage(input_tokens=1, output_tokens=1)),
        ])
        types = [_parse_sse(e)[0] for e in collected]
        assert "content_block_stop" in types
        assert types.index("content_block_stop") < types.index("message_delta")

    @pytest.mark.asyncio
    async def test_empty_stream_still_closes_grammar(self):
        collected = await _collect([])
        types = [_parse_sse(e)[0] for e in collected]
        assert types == ["message_start", "ping", "message_delta", "message_stop"]


class TestSafeguardResults:
    @pytest.mark.asyncio
    async def test_results_on_message_delta(self):
        results = [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {"tu_1": {"type": "evaluated", "outcome": "not_flagged"}}}}]
        collected = await _collect([
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="tool_use",
                          usage=InternalUsage(input_tokens=1, output_tokens=1), safeguard_results=results),
        ])
        delta = _parse_sse(next(e for e in collected if e.startswith("event: message_delta")))[1]["delta"]
        assert delta["safeguard_results"] == results

    @pytest.mark.asyncio
    async def test_absent_results_not_emitted(self):
        collected = await _collect([
            InternalChunk(kind=ChunkKind.MESSAGE_STOP, stop_reason="end_turn", usage=InternalUsage(input_tokens=1, output_tokens=1)),
        ])
        delta = _parse_sse(next(e for e in collected if e.startswith("event: message_delta")))[1]["delta"]
        assert "safeguard_results" not in delta
