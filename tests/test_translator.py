"""Tests for Anthropic <-> LangChain message translation."""

from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ttllm.core.translator import (
    extract_invoke_params,
    from_langchain_response,
    to_langchain_messages,
)
from ttllm.schemas.anthropic import (
    Message,
    MessagesRequest,
    TextBlock,
    ToolUseBlock,
)


def _make_request(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "test-model",
        "max_tokens": 100,
        "messages": [Message(role="user", content="Hello")],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


class TestToLangchainMessages:
    def test_simple_text_message(self):
        request = _make_request()
        msgs = to_langchain_messages(request)
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == "Hello"

    def test_with_system_message(self):
        request = _make_request(system="You are helpful.")
        msgs = to_langchain_messages(request)
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "You are helpful."
        assert isinstance(msgs[1], HumanMessage)

    def test_system_as_text_blocks(self):
        request = _make_request(
            system=[TextBlock(text="Part 1"), TextBlock(text="Part 2")]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], SystemMessage)
        assert "Part 1" in msgs[0].content
        assert "Part 2" in msgs[0].content

    def test_multi_turn_conversation(self):
        request = _make_request(
            messages=[
                Message(role="user", content="Hi"),
                Message(role="assistant", content="Hello!"),
                Message(role="user", content="How are you?"),
            ]
        )
        msgs = to_langchain_messages(request)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)

    def test_content_blocks(self):
        request = _make_request(
            messages=[
                Message(
                    role="user",
                    content=[TextBlock(text="Hello"), TextBlock(text="World")],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert len(msgs) == 1
        content = msgs[0].content
        assert isinstance(content, list)
        assert len(content) == 2

    def test_assistant_with_tool_use(self):
        request = _make_request(
            messages=[
                Message(role="user", content="Search for X"),
                Message(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search."),
                        ToolUseBlock(
                            id="tu_1", name="search", input={"query": "X"}
                        ),
                    ],
                ),
            ]
        )
        msgs = to_langchain_messages(request)
        assert len(msgs) == 2
        ai_msg = msgs[1]
        assert isinstance(ai_msg, AIMessage)
        assert ai_msg.content == "Let me search."
        assert len(ai_msg.tool_calls) == 1
        assert ai_msg.tool_calls[0]["name"] == "search"


class TestExtractInvokeParams:
    def test_basic(self):
        request = _make_request(max_tokens=200)
        params = extract_invoke_params(request)
        assert params["max_tokens"] == 200

    def test_with_optional_params(self):
        request = _make_request(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
        )
        params = extract_invoke_params(request)
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["top_k"] == 40
        assert params["stop"] == ["END"]

    def test_none_params_excluded(self):
        request = _make_request()
        params = extract_invoke_params(request)
        assert "temperature" not in params
        assert "top_p" not in params


class TestFromLangchainResponse:
    def test_simple_text(self):
        ai_msg = AIMessage(content="Hello there!")
        request_id = uuid.uuid4()
        response = from_langchain_response(ai_msg, "test-model", request_id)

        assert response.model == "test-model"
        assert response.role == "assistant"
        assert len(response.content) == 1
        assert response.content[0].type == "text"
        assert response.content[0].text == "Hello there!"
        assert response.id.startswith("msg_")

    def test_with_usage_metadata(self):
        ai_msg = AIMessage(content="Hi")
        ai_msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        request_id = uuid.uuid4()
        response = from_langchain_response(ai_msg, "test-model", request_id)

        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_with_tool_calls(self):
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [
            {"id": "tc_1", "name": "search", "args": {"q": "test"}}
        ]
        request_id = uuid.uuid4()
        response = from_langchain_response(ai_msg, "test-model", request_id)

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "search"
        assert response.stop_reason == "tool_use"
