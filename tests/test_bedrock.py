"""Tests for direct Bedrock Converse integration."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from ttllm.core.bedrock import (
    build_converse_request,
    parse_converse_response,
)
from ttllm.schemas.anthropic import (
    DocumentBlock,
    DocumentSource,
    ImageBlock,
    ImageSource,
    Message,
    MessagesRequest,
    ServerToolDefinition,
    TextBlock,
    ThinkingBlock,
    ToolChoiceAuto,
    ToolDefinition,
    ToolInputSchema,
    ToolResultBlock,
    ToolUseBlock,
)


def _make_model(**overrides):
    model = MagicMock()
    model.provider = "bedrock"
    model.provider_model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
    model.name = "claude-sonnet"
    model.config_json = {"region": "us-east-1"}
    for k, v in overrides.items():
        setattr(model, k, v)
    return model


def _make_request(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-sonnet",
        "max_tokens": 1024,
        "messages": [Message(role="user", content="Hello")],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


class TestBuildConverseRequest:
    def test_basic_text_message(self):
        request = _make_request()
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["modelId"] == "anthropic.claude-sonnet-4-20250514-v1:0"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == [{"text": "Hello"}]
        assert result["inferenceConfig"]["maxTokens"] == 1024

    def test_system_message_string(self):
        request = _make_request(system="Be helpful.")
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Be helpful."}]

    def test_system_message_blocks(self):
        request = _make_request(system=[TextBlock(text="Part 1"), TextBlock(text="Part 2")])
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Part 1"}, {"text": "Part 2"}]

    def test_inference_config(self):
        request = _make_request(temperature=0.7, top_p=0.9, stop_sequences=["END"])
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["inferenceConfig"]["temperature"] == 0.7
        assert result["inferenceConfig"]["topP"] == 0.9
        assert result["inferenceConfig"]["stopSequences"] == ["END"]

    def test_top_k_in_additional_fields(self):
        request = _make_request(top_k=50)
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["additionalModelRequestFields"]["top_k"] == 50

    def test_thinking_config_passthrough(self):
        request = _make_request(thinking={"type": "enabled", "budget_tokens": 5000})
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["additionalModelRequestFields"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 5000,
        }

    def test_tool_definitions(self):
        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                input_schema=ToolInputSchema(
                    properties={"query": {"type": "string"}},
                    required=["query"],
                ),
            )
        ]
        request = _make_request(tools=tools, tool_choice=ToolChoiceAuto())
        model = _make_model()
        result = build_converse_request(request, model)

        assert "toolConfig" in result
        tool_specs = result["toolConfig"]["tools"]
        assert len(tool_specs) == 1
        assert tool_specs[0]["toolSpec"]["name"] == "search"
        assert tool_specs[0]["toolSpec"]["inputSchema"]["json"]["properties"] == {"query": {"type": "string"}}
        assert result["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_server_tools_filtered_from_tool_config(self):
        tools = [
            ToolDefinition(
                name="search",
                description="Search",
                input_schema=ToolInputSchema(),
            ),
            ServerToolDefinition(type="web_search_20250305", name="web_search"),
        ]
        request = _make_request(tools=tools)
        model = _make_model()
        result = build_converse_request(request, model)

        tool_specs = result["toolConfig"]["tools"]
        assert len(tool_specs) == 1
        assert tool_specs[0]["toolSpec"]["name"] == "search"

    def test_image_block_conversion(self):
        messages = [
            Message(
                role="user",
                content=[
                    TextBlock(text="What is this?"),
                    ImageBlock(source=ImageSource(media_type="image/png", data="aWdub3Jl")),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0] == {"text": "What is this?"}
        assert "image" in content[1]
        assert content[1]["image"]["format"] == "png"

    def test_tool_use_block_conversion(self):
        messages = [
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(id="tu_123", name="search", input={"q": "test"}),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["toolUse"]["toolUseId"] == "tu_123"
        assert content[0]["toolUse"]["name"] == "search"
        assert content[0]["toolUse"]["input"] == {"q": "test"}

    def test_tool_result_block_conversion(self):
        messages = [
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="tu_123", content="Result text", is_error=False),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["toolResult"]["toolUseId"] == "tu_123"
        assert content[0]["toolResult"]["content"] == [{"text": "Result text"}]
        assert content[0]["toolResult"]["status"] == "success"

    def test_tool_result_error(self):
        messages = [
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="tu_456", content="Failed", is_error=True),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["toolResult"]["status"] == "error"

    def test_thinking_block_conversion(self):
        messages = [
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Let me think about this...", signature="sig_abc"),
                    TextBlock(text="Here is my answer."),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["reasoningContent"]["reasoningText"]["text"] == "Let me think about this..."
        assert content[0]["reasoningContent"]["reasoningText"]["signature"] == "sig_abc"
        assert content[1] == {"text": "Here is my answer."}

    def test_document_block_conversion(self):
        messages = [
            Message(
                role="user",
                content=[
                    DocumentBlock(
                        source=DocumentSource(media_type="application/pdf", data="SlZCRVI="),
                        title="report.pdf",
                    ),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["document"]["format"] == "pdf"
        assert content[0]["document"]["name"] == "report.pdf"


class TestParseConverseResponse:
    def test_basic_text_response(self):
        response = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert result.id == f"msg_{request_id.hex[:24]}"
        assert result.model == "claude-sonnet"
        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_tool_use_response(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "I'll search for that."},
                        {"toolUse": {"toolUseId": "tu_abc", "name": "search", "input": {"q": "test"}}},
                    ]
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 20, "outputTokens": 15},
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert result.stop_reason == "tool_use"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "tool_use"
        assert result.content[1].id == "tu_abc"
        assert result.content[1].name == "search"

    def test_thinking_block_response(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"reasoningContent": {"reasoningText": {"text": "Thinking deeply...", "signature": "sig_xyz"}}},
                        {"text": "My conclusion."},
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 30, "outputTokens": 50},
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].thinking == "Thinking deeply..."
        assert result.content[0].signature == "sig_xyz"
        assert result.content[1].type == "text"

    def test_cache_token_mapping(self):
        response = {
            "output": {"message": {"content": [{"text": "cached"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 100,
                "outputTokens": 20,
                "cacheReadInputTokens": 50,
                "cacheWriteInputTokens": 30,
            },
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert result.usage.input_tokens == 150
        assert result.usage.output_tokens == 20
        assert result.usage.cache_read_input_tokens == 50
        assert result.usage.cache_creation_input_tokens == 30

    def test_max_tokens_stop_reason(self):
        response = {
            "output": {"message": {"content": [{"text": "truncated..."}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 10, "outputTokens": 4096},
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert result.stop_reason == "max_tokens"

    def test_empty_content_gets_empty_text_block(self):
        response = {
            "output": {"message": {"content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 0},
        }
        request_id = uuid.uuid4()
        result = parse_converse_response(response, "claude-sonnet", request_id)

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == ""


class TestServerToolGating:
    def test_server_tool_error_raised(self):
        from ttllm.core.gateway import ServerToolError, _has_server_tools

        tools = [
            ToolDefinition(name="search", description="", input_schema=ToolInputSchema()),
            ServerToolDefinition(type="web_search_20250305", name="web_search"),
        ]
        request = _make_request(tools=tools)
        assert _has_server_tools(request) is True

    def test_no_server_tools(self):
        from ttllm.core.gateway import _has_server_tools

        tools = [
            ToolDefinition(name="search", description="", input_schema=ToolInputSchema()),
        ]
        request = _make_request(tools=tools)
        assert _has_server_tools(request) is False

    def test_none_tools(self):
        from ttllm.core.gateway import _has_server_tools

        request = _make_request(tools=None)
        assert _has_server_tools(request) is False


class TestInvokeConverse:
    @pytest.mark.asyncio
    async def test_invoke_calls_boto3(self):
        from ttllm.core.bedrock import invoke_converse

        model = _make_model()
        request = _make_request()
        request_id = uuid.uuid4()

        mock_response = {
            "output": {"message": {"content": [{"text": "Hi there!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 8, "outputTokens": 4},
        }

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await invoke_converse(request, model, request_id)

        assert result.content[0].text == "Hi there!"
        assert result.usage.input_tokens == 8
        mock_client.converse.assert_called_once()
