"""Tests for LangChainProvider's native-chunk -> internal-chunk translation."""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from ttllm.core.model import ChunkKind
from ttllm.core.providers.langchain.provider import stream_chunks


async def _lc_stream(chunks):
    for c in chunks:
        yield c


class _FakeState:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.stop_reason = ""
        self.text = ""
        self.tool_calls: list[dict] = []


async def _collect(chunks, state=None):
    return [c async for c in stream_chunks(_lc_stream(chunks), state=state)]


class TestStreamChunks:
    async def test_text_only_stream(self):
        chunks = [AIMessageChunk(content="Hello"), AIMessageChunk(content=" world")]
        collected = await _collect(chunks)
        kinds = [c.kind for c in collected]
        assert kinds[0] == ChunkKind.BLOCK_START
        assert kinds[1] == ChunkKind.TEXT_DELTA
        assert kinds[2] == ChunkKind.TEXT_DELTA
        assert kinds[-2] == ChunkKind.BLOCK_STOP
        assert kinds[-1] == ChunkKind.MESSAGE_STOP
        text = "".join(c.text for c in collected if c.kind == ChunkKind.TEXT_DELTA)
        assert text == "Hello world"
        assert collected[-1].stop_reason == "end_turn"

    async def test_tool_use_only_stream(self):
        chunks = [
            AIMessageChunk(content="", tool_call_chunks=[{"id": "tu_1", "name": "search", "args": '{"q":', "index": 0}]),
            AIMessageChunk(content="", tool_call_chunks=[{"id": None, "name": None, "args": '"x"}', "index": 0}]),
        ]
        collected = await _collect(chunks)
        start = next(c for c in collected if c.kind == ChunkKind.BLOCK_START)
        assert start.part.type == "tool_call"
        assert start.part.name == "search"
        partials = [c.partial_json for c in collected if c.kind == ChunkKind.TOOL_ARGS_DELTA]
        assert "".join(partials) == '{"q":"x"}'
        stop = next(c for c in collected if c.kind == ChunkKind.MESSAGE_STOP)
        assert stop.stop_reason == "tool_use"

    async def test_text_then_tool_use_closes_text_block_first(self):
        chunks = [
            AIMessageChunk(content="Let me check."),
            AIMessageChunk(content="", tool_call_chunks=[{"id": "tu_1", "name": "search", "args": "{}", "index": 0}]),
        ]
        collected = await _collect(chunks)
        kinds = [c.kind for c in collected]
        # text block start/delta/stop, then tool block start/delta/stop, then message_stop
        assert kinds == [
            ChunkKind.BLOCK_START, ChunkKind.TEXT_DELTA, ChunkKind.BLOCK_STOP,
            ChunkKind.BLOCK_START, ChunkKind.TOOL_ARGS_DELTA, ChunkKind.BLOCK_STOP,
            ChunkKind.MESSAGE_STOP,
        ]
        assert collected[0].index == 0
        assert collected[3].index == 1

    async def test_multiple_sequential_tool_calls_get_distinct_indices(self):
        chunks = [
            AIMessageChunk(content="", tool_call_chunks=[{"id": "tu_1", "name": "a", "args": "{}", "index": 0}]),
            AIMessageChunk(content="", tool_call_chunks=[{"id": "tu_2", "name": "b", "args": "{}", "index": 1}]),
        ]
        collected = await _collect(chunks)
        starts = [c for c in collected if c.kind == ChunkKind.BLOCK_START]
        assert len(starts) == 2
        assert starts[0].index != starts[1].index
        stops = [c for c in collected if c.kind == ChunkKind.BLOCK_STOP]
        assert len(stops) == 2

    async def test_usage_metadata_propagated(self):
        chunk = AIMessageChunk(content="hi")
        chunk.usage_metadata = {
            "input_tokens": 10, "output_tokens": 5,
            "input_token_details": {"cache_read": 3},
        }
        collected = await _collect([chunk])
        stop = next(c for c in collected if c.kind == ChunkKind.MESSAGE_STOP)
        assert stop.usage.input_tokens == 10
        assert stop.usage.output_tokens == 5
        assert stop.usage.cache_read_tokens == 3

    async def test_state_accumulates_full_response(self):
        state = _FakeState()
        chunks = [AIMessageChunk(content="Hi"), AIMessageChunk(content=" there")]
        await _collect(chunks, state=state)
        assert state.text == "Hi there"
        assert state.stop_reason == "end_turn"
