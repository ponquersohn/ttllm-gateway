"""Tests for SSE streaming with tool-use support."""

from __future__ import annotations

import json
import uuid

import pytest
from langchain_core.messages import AIMessageChunk

from ttllm.core.streaming import format_sse_stream


def _parse_events(raw_events: list[str]) -> list[tuple[str, dict]]:
    """Parse SSE event strings into (event_type, data) tuples."""
    parsed = []
    for event_str in raw_events:
        lines = event_str.strip().split("\n")
        event_type = lines[0].replace("event: ", "")
        data = json.loads(lines[1].replace("data: ", ""))
        parsed.append((event_type, data))
    return parsed


async def _collect_stream(stream) -> list[str]:
    events = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_text_only_stream():
    async def mock_stream():
        yield AIMessageChunk(content="Hello ")
        yield AIMessageChunk(content="world")

    token_usage: dict[str, int] = {}
    request_id = uuid.uuid4()
    stream = format_sse_stream(mock_stream(), "test-model", request_id, token_usage)
    raw = await _collect_stream(stream)
    events = _parse_events(raw)

    event_types = [e[0] for e in events]
    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "content_block_delta" in event_types
    assert "content_block_stop" in event_types
    assert "message_delta" in event_types
    assert "message_stop" in event_types

    # Check stop reason is end_turn for text-only
    delta_event = next(e for e in events if e[0] == "message_delta")
    assert delta_event[1]["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_tool_use_stream():
    async def mock_stream():
        # First chunk: text
        yield AIMessageChunk(content="Let me search.")
        # Tool call start chunk
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"id": "tc_123", "name": "search", "args": '{"q": ', "index": 0}
            ],
        )
        # Tool call continuation chunk
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"id": None, "name": None, "args": '"weather"}', "index": 0}
            ],
        )

    token_usage: dict[str, int] = {}
    request_id = uuid.uuid4()
    stream = format_sse_stream(mock_stream(), "test-model", request_id, token_usage)
    raw = await _collect_stream(stream)
    events = _parse_events(raw)

    event_types = [e[0] for e in events]

    # Should have two content_block_start events (text + tool_use)
    block_starts = [e for e in events if e[0] == "content_block_start"]
    assert len(block_starts) == 2
    assert block_starts[0][1]["content_block"]["type"] == "text"
    assert block_starts[1][1]["content_block"]["type"] == "tool_use"
    assert block_starts[1][1]["content_block"]["id"] == "tc_123"
    assert block_starts[1][1]["content_block"]["name"] == "search"

    # Should have text_delta and input_json_delta
    deltas = [e for e in events if e[0] == "content_block_delta"]
    text_deltas = [d for d in deltas if d[1]["delta"]["type"] == "text_delta"]
    json_deltas = [d for d in deltas if d[1]["delta"]["type"] == "input_json_delta"]
    assert len(text_deltas) >= 1
    assert text_deltas[0][1]["delta"]["text"] == "Let me search."
    assert len(json_deltas) == 2
    assert json_deltas[0][1]["delta"]["partial_json"] == '{"q": '
    assert json_deltas[1][1]["delta"]["partial_json"] == '"weather"}'

    # Stop reason should be tool_use
    delta_event = next(e for e in events if e[0] == "message_delta")
    assert delta_event[1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_tool_use_only_stream():
    """Tool use without preceding text content."""

    async def mock_stream():
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"id": "tc_abc", "name": "get_weather", "args": '{"city":', "index": 0}
            ],
        )
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"id": None, "name": None, "args": ' "NYC"}', "index": 0}
            ],
        )

    token_usage: dict[str, int] = {}
    request_id = uuid.uuid4()
    stream = format_sse_stream(mock_stream(), "test-model", request_id, token_usage)
    raw = await _collect_stream(stream)
    events = _parse_events(raw)

    block_starts = [e for e in events if e[0] == "content_block_start"]
    # Only tool_use block, no text block
    assert len(block_starts) == 1
    assert block_starts[0][1]["content_block"]["type"] == "tool_use"
    assert block_starts[0][1]["content_block"]["name"] == "get_weather"

    delta_event = next(e for e in events if e[0] == "message_delta")
    assert delta_event[1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_message_delta_reports_input_tokens():
    async def mock_stream():
        chunk = AIMessageChunk(content="Hi")
        chunk.usage_metadata = {"input_tokens": 42, "output_tokens": 7}
        yield chunk

    request_id = uuid.uuid4()
    stream = format_sse_stream(mock_stream(), "test-model", request_id, {})
    events = _parse_events(await _collect_stream(stream))

    usage = next(e for e in events if e[0] == "message_delta")[1]["usage"]
    assert usage["input_tokens"] == 42
    assert usage["output_tokens"] == 7
    # No cache info supplied → cache_read_input_tokens is null
    assert usage["cache_read_input_tokens"] is None


@pytest.mark.asyncio
async def test_message_delta_surfaces_cache_read():
    async def mock_stream():
        chunk = AIMessageChunk(content="Hi")
        chunk.usage_metadata = {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_token_details": {"cache_read": 60},
        }
        yield chunk

    request_id = uuid.uuid4()
    stream = format_sse_stream(mock_stream(), "test-model", request_id, {})
    events = _parse_events(await _collect_stream(stream))

    usage = next(e for e in events if e[0] == "message_delta")[1]["usage"]
    assert usage["input_tokens"] == 100
    assert usage["cache_read_input_tokens"] == 60
