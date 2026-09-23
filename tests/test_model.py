"""Tests for the internal representation types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ttllm.core.model import (
    ChunkKind,
    DocumentPart,
    ImagePart,
    InternalChunk,
    InternalMessage,
    InternalRequest,
    Part,
    RedactedThinkingPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from pydantic import TypeAdapter

_PartAdapter = TypeAdapter(Part)


class TestPartDiscrimination:
    @pytest.mark.parametrize(
        "payload,expected_type",
        [
            ({"type": "text", "text": "hi"}, TextPart),
            ({"type": "image", "media_type": "image/png", "data": "aa"}, ImagePart),
            ({"type": "document", "media_type": "application/pdf", "data": "aa"}, DocumentPart),
            ({"type": "tool_call", "id": "t1", "name": "search", "input": {}}, ToolCallPart),
            ({"type": "tool_result", "tool_call_id": "t1", "content": []}, ToolResultPart),
            ({"type": "thinking", "text": "hmm", "signature": "sig"}, ThinkingPart),
            ({"type": "redacted_thinking", "data": "xx"}, RedactedThinkingPart),
        ],
    )
    def test_discriminates_by_type(self, payload, expected_type):
        part = _PartAdapter.validate_python(payload)
        assert isinstance(part, expected_type)

    def test_tool_result_content_only_accepts_text_or_image(self):
        with pytest.raises(ValidationError):
            ToolResultPart(tool_call_id="t1", content=[{"type": "tool_call", "id": "x", "name": "y", "input": {}}])


class TestInternalRequest:
    def test_defaults(self):
        request = InternalRequest(
            provider_model_id="m",
            max_tokens=100,
            messages=[InternalMessage(role="user", content=[TextPart(text="hi")])],
        )
        assert request.system is None
        assert request.tools == []
        assert request.tool_choice is None
        assert request.thinking is None
        assert request.stop_sequences == []

    def test_message_role_rejects_system(self):
        with pytest.raises(ValidationError):
            InternalMessage(role="system", content=[TextPart(text="hi")])


class TestInternalChunk:
    def test_block_start_carries_a_part(self):
        chunk = InternalChunk(kind=ChunkKind.BLOCK_START, index=0, part=TextPart(text=""))
        assert chunk.part.type == "text"

    def test_message_stop_carries_usage(self):
        from ttllm.core.model import InternalUsage

        chunk = InternalChunk(
            kind=ChunkKind.MESSAGE_STOP,
            stop_reason="end_turn",
            usage=InternalUsage(input_tokens=1, output_tokens=2),
        )
        assert chunk.usage.input_tokens == 1
