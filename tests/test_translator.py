"""Tests for the internal representation <-> LangChain messages translator."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ttllm.core.errors import ServerToolError, UnsupportedContentError
from ttllm.core.model import (
    DocumentPart,
    ImagePart,
    InternalMessage,
    InternalRequest,
    RedactedThinkingPart,
    ServerToolCallPart,
    ServerToolResultPart,
    ServerToolSpec,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
)
from ttllm.core.providers.langchain.translation import (
    bind_tools_to_model,
    convert_tool_choice,
    extract_invoke_params,
    from_langchain_response,
    to_langchain_messages,
)


def _make_request(**kwargs) -> InternalRequest:
    defaults = {
        "provider_model_id": "gpt-4o",
        "max_tokens": 1024,
        "messages": [InternalMessage(role="user", content=[TextPart(text="Hello")])],
    }
    defaults.update(kwargs)
    return InternalRequest(**defaults)


class TestToLangchainMessages:
    def test_simple_text_message(self):
        msgs = to_langchain_messages(_make_request())
        assert len(msgs) == 1
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == [{"type": "text", "text": "Hello"}]

    def test_with_system_message(self):
        msgs = to_langchain_messages(_make_request(system=[TextPart(text="Be helpful.")]))
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "Be helpful."

    def test_multi_turn_conversation(self):
        request = _make_request(
            messages=[
                InternalMessage(role="user", content=[TextPart(text="Hi")]),
                InternalMessage(role="assistant", content=[TextPart(text="Hello!")]),
                InternalMessage(role="user", content=[TextPart(text="How are you?")]),
            ]
        )
        msgs = to_langchain_messages(request)
        assert len(msgs) == 3
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)
        assert isinstance(msgs[2], HumanMessage)

    def test_image_content(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="user",
                    content=[TextPart(text="what's this?"), ImagePart(media_type="image/png", data="aWdub3Jl")],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert msgs[0].content[1]["type"] == "image_url"
        assert "data:image/png;base64,aWdub3Jl" in msgs[0].content[1]["image_url"]["url"]

    def test_assistant_with_tool_use(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="assistant",
                    content=[TextPart(text="Let me check."), ToolCallPart(id="tu_1", name="search", input={"q": "x"})],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].tool_calls == [{"name": "search", "args": {"q": "x"}, "id": "tu_1"}]
        assert msgs[0].content == "Let me check."

    def test_tool_result_becomes_tool_message(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="user",
                    content=[ToolResultPart(tool_call_id="tu_1", content=[TextPart(text="42")])],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].content == "42"
        assert msgs[0].tool_call_id == "tu_1"

    def test_tool_result_with_image_is_emulated_not_dropped(self):
        """Previously this crashed (AttributeError on ImageBlock.text). Now it's
        passed through as multimodal ToolMessage content instead of erroring or
        silently dropping the image."""
        request = _make_request(
            messages=[
                InternalMessage(
                    role="user",
                    content=[
                        ToolResultPart(
                            tool_call_id="tu_1",
                            content=[TextPart(text="see:"), ImagePart(media_type="image/png", data="aWdub3Jl")],
                        )
                    ],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], ToolMessage)
        assert msgs[0].content == [
            {"type": "text", "text": "see:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aWdub3Jl"}},
        ]

    def test_tool_result_and_other_content_split(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="user",
                    content=[
                        ToolResultPart(tool_call_id="tu_1", content=[TextPart(text="42")]),
                        TextPart(text="thanks"),
                    ],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], ToolMessage)
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == [{"type": "text", "text": "thanks"}]

    def test_document_raises_unsupported(self):
        request = _make_request(
            messages=[
                InternalMessage(role="user", content=[DocumentPart(media_type="application/pdf", data="AA==")]),
            ]
        )
        with pytest.raises(UnsupportedContentError):
            to_langchain_messages(request)

    def test_document_in_assistant_turn_raises_unsupported(self):
        request = _make_request(
            messages=[
                InternalMessage(role="assistant", content=[DocumentPart(media_type="application/pdf", data="AA==")]),
            ]
        )
        with pytest.raises(UnsupportedContentError):
            to_langchain_messages(request)

    def test_server_tools_raise_server_tool_error(self):
        # Whether a provider can proxy server-side tools is a provider decision -- the
        # adapter converts them faithfully; LangChain (which can't proxy them) rejects
        # here, at translation time.
        request = _make_request(server_tools=[ServerToolSpec(type="web_search_20250305", name="web_search")])
        with pytest.raises(ServerToolError):
            to_langchain_messages(request)

    def test_server_tool_call_in_history_raises(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="assistant",
                    content=[ServerToolCallPart(id="st_1", name="web_search", input={"q": "cats"})],
                )
            ]
        )
        with pytest.raises(ServerToolError):
            to_langchain_messages(request)

    def test_server_tool_result_in_user_turn_raises(self):
        request = _make_request(
            messages=[
                InternalMessage(role="user", content=[ServerToolResultPart(tool_call_id="st_1", content=[])]),
            ]
        )
        with pytest.raises(ServerToolError):
            to_langchain_messages(request)

    def test_thinking_in_assistant_turn_emulated_as_text(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="assistant",
                    content=[ThinkingPart(text="pondering...", signature="sig"), TextPart(text="answer")],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == "pondering...\nanswer"

    def test_redacted_thinking_in_assistant_turn_silently_dropped(self):
        request = _make_request(
            messages=[
                InternalMessage(
                    role="assistant",
                    content=[RedactedThinkingPart(data="encrypted"), TextPart(text="answer")],
                )
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], AIMessage)
        assert msgs[0].content == "answer"

    def test_tool_call_in_malformed_position_becomes_text_stub(self):
        # A tool_call in a non-assistant turn shouldn't occur in a valid Anthropic
        # conversation; best-effort text stub, not a real feature gap.
        request = _make_request(
            messages=[
                InternalMessage(role="user", content=[ToolCallPart(id="tu_1", name="search", input={"q": "x"})]),
            ]
        )
        msgs = to_langchain_messages(request)
        assert isinstance(msgs[0], HumanMessage)
        assert msgs[0].content == [{"type": "text", "text": "[tool_use: search({'q': 'x'})]"}]


class TestExtractInvokeParams:
    def test_basic(self):
        params = extract_invoke_params(_make_request())
        assert params == {"max_tokens": 1024}

    def test_with_optional_params(self):
        request = _make_request(temperature=0.5, top_p=0.9, top_k=40, stop_sequences=["STOP"])
        params = extract_invoke_params(request)
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9
        assert params["top_k"] == 40
        assert params["stop"] == ["STOP"]

    def test_none_params_excluded(self):
        params = extract_invoke_params(_make_request())
        assert "temperature" not in params
        assert "top_p" not in params
        assert "top_k" not in params
        assert "stop" not in params


class TestFromLangchainResponse:
    def test_simple_text(self):
        response = AIMessage(content="Hello there!")
        result = from_langchain_response(response, input_tokens=5, output_tokens=3)
        assert result.content[0].text == "Hello there!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 3

    def test_with_usage_metadata(self):
        response = AIMessage(content="Hi")
        response.usage_metadata = {"input_tokens": 10, "output_tokens": 2}
        result = from_langchain_response(response)
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 2

    def test_with_tool_calls(self):
        response = AIMessage(content="")
        response.tool_calls = [{"id": "tu_1", "name": "search", "args": {"q": "x"}}]
        result = from_langchain_response(response)
        assert result.stop_reason == "tool_use"
        assert result.content[0].type == "tool_call"
        assert result.content[0].name == "search"

    def test_content_filter_surfaced_as_refusal(self):
        """The bug this guards against: a moderation/guardrail cutoff previously
        stayed "end_turn" (the fallback default), indistinguishable from a normal
        completion -- the caller had no way to know the model's output was blocked."""
        response = AIMessage(content="")
        response.response_metadata = {"finish_reason": "content_filter"}
        result = from_langchain_response(response)
        assert result.stop_reason == "refusal"


class TestConvertToolChoice:
    def test_none(self):
        assert convert_tool_choice(None) is None

    def test_auto(self):
        assert convert_tool_choice(ToolChoice(mode="auto")) == "auto"

    def test_any(self):
        assert convert_tool_choice(ToolChoice(mode="any")) == "any"

    def test_specific_tool(self):
        assert convert_tool_choice(ToolChoice(mode="tool", tool_name="search")) == "search"

    def test_none_mode(self):
        assert convert_tool_choice(ToolChoice(mode="none")) is None


class TestBindToolsToModel:
    def test_no_tools_returns_model_unchanged(self):
        model = object()
        result = bind_tools_to_model(model, None, None)
        assert result is model

    def test_empty_tools_returns_model_unchanged(self):
        model = object()
        result = bind_tools_to_model(model, [], None)
        assert result is model

    def test_with_tools_calls_bind_tools(self):
        class FakeModel:
            def bind_tools(self, tools, **kwargs):
                self.bound_tools = tools
                self.bound_kwargs = kwargs
                return "bound"

        model = FakeModel()
        tools = [ToolSpec(name="search", description="Search", input_schema={"type": "object", "properties": {}})]
        result = bind_tools_to_model(model, tools, ToolChoice(mode="auto"))
        assert result == "bound"
        assert model.bound_tools[0]["name"] == "search"
        assert model.bound_kwargs == {"tool_choice": "auto"}

    def test_with_tools_no_tool_choice(self):
        class FakeModel:
            def bind_tools(self, tools, **kwargs):
                self.bound_kwargs = kwargs
                return "bound"

        model = FakeModel()
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        bind_tools_to_model(model, tools, None)
        assert model.bound_kwargs == {}
