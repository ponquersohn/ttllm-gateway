"""Tests for direct Bedrock Converse integration."""

from __future__ import annotations

import json
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
    RedactedThinkingBlock,
    ServerToolDefinition,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceNone,
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

    def test_mid_conversation_system_message_lifted_to_system(self):
        # Anthropic mid-conversation system messages arrive as a system-role turn
        # inside messages; Bedrock Converse has no inline system role, so it must
        # be lifted out into the top-level system array.
        request = _make_request(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="system", content="Terse mode enabled."),
                Message(role="user", content="Continue"),
            ]
        )
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Terse mode enabled."}]
        assert [m["role"] for m in result["messages"]] == ["user", "user"]

    def test_mid_conversation_system_appended_after_top_level_system(self):
        request = _make_request(
            system="Base prompt.",
            messages=[
                Message(role="user", content="Hi"),
                Message(role="system", content="Switch to JSON output."),
            ],
        )
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Base prompt."}, {"text": "Switch to JSON output."}]
        assert [m["role"] for m in result["messages"]] == ["user"]

    def test_system_message_block_content_flattened(self):
        request = _make_request(
            messages=[
                Message(role="user", content="Hi"),
                Message(
                    role="system",
                    content=[TextBlock(text="Line A"), TextBlock(text="Line B")],
                ),
            ]
        )
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Line A\nLine B"}]

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
        result, cache_read, cache_write = parse_converse_response(response, "claude-sonnet", request_id)

        assert cache_read == 0
        assert cache_write == 0
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
        result, _, _ = parse_converse_response(response, "claude-sonnet", request_id)

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
        result, _, _ = parse_converse_response(response, "claude-sonnet", request_id)

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
        result, cache_read, cache_write = parse_converse_response(response, "claude-sonnet", request_id)

        # cache_read is reported separately, NOT folded into billed input_tokens.
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 20
        assert result.usage.cache_read_input_tokens == 50
        assert result.usage.cache_creation_input_tokens == 30
        assert cache_read == 50
        assert cache_write == 30

    def test_max_tokens_stop_reason(self):
        response = {
            "output": {"message": {"content": [{"text": "truncated..."}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 10, "outputTokens": 4096},
        }
        request_id = uuid.uuid4()
        result, _, _ = parse_converse_response(response, "claude-sonnet", request_id)

        assert result.stop_reason == "max_tokens"

    def test_empty_content_gets_empty_text_block(self):
        response = {
            "output": {"message": {"content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 0},
        }
        request_id = uuid.uuid4()
        result, _, _ = parse_converse_response(response, "claude-sonnet", request_id)

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

            result, cache_read, cache_write = await invoke_converse(request, model, request_id)

        assert result.content[0].text == "Hi there!"
        assert result.usage.input_tokens == 8
        assert cache_read == 0
        assert cache_write == 0
        mock_client.converse.assert_called_once()


class TestGetBoto3Client:
    def test_endpoint_url_passed_to_client(self):
        from ttllm.core import bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={
            "region": "us-east-1",
            "endpoint_url": "http://fake-bedrock:9099",
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
        })

        with patch("ttllm.core.bedrock.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        mock_session.return_value.client.assert_called_once()
        args, kwargs = mock_session.return_value.client.call_args
        assert args[0] == "bedrock-runtime"
        assert kwargs["endpoint_url"] == "http://fake-bedrock:9099"

    def test_no_endpoint_url_omits_kwarg(self):
        from ttllm.core import bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={"region": "us-east-1"})

        with patch("ttllm.core.bedrock.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        assert "endpoint_url" not in kwargs

    def test_default_timeouts_applied(self):
        from ttllm.core import bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={"region": "us-east-1"})

        with patch("ttllm.core.bedrock.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        boto_config = kwargs["config"]
        assert boto_config.read_timeout == bedrock._DEFAULT_READ_TIMEOUT
        assert boto_config.connect_timeout == bedrock._DEFAULT_CONNECT_TIMEOUT
        assert boto_config.retries == {"mode": "standard", "max_attempts": bedrock._DEFAULT_MAX_ATTEMPTS}

    def test_config_json_timeout_overrides(self):
        from ttllm.core import bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={
            "region": "us-east-1",
            "read_timeout": 600,
            "connect_timeout": 5,
            "retry_max_attempts": 1,
        })

        with patch("ttllm.core.bedrock.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        boto_config = kwargs["config"]
        assert boto_config.read_timeout == 600
        assert boto_config.connect_timeout == 5
        assert boto_config.retries == {"mode": "standard", "max_attempts": 1}

    def test_distinct_timeouts_get_distinct_cached_clients(self):
        from ttllm.core import bedrock

        bedrock._CLIENT_CACHE.clear()
        base = {"region": "us-east-1"}
        model_default = _make_model(config_json=base)
        model_tuned = _make_model(config_json={**base, "read_timeout": 600})

        with patch("ttllm.core.bedrock.boto3.Session") as mock_session:
            # A fresh mock per client() call, so identity reflects cache behavior.
            mock_session.return_value.client.side_effect = lambda *a, **k: MagicMock()
            client_a = bedrock.get_boto3_client(model_default)
            client_b = bedrock.get_boto3_client(model_tuned)
            client_a_again = bedrock.get_boto3_client(model_default)

        assert client_a is not client_b
        assert client_a is client_a_again


class TestToolChoice:
    def test_tool_choice_none_omits_tool_config(self):
        tools = [ToolDefinition(name="search", description="", input_schema=ToolInputSchema())]
        request = _make_request(tools=tools, tool_choice=ToolChoiceNone())
        result = build_converse_request(request, _make_model())

        assert "toolConfig" not in result

    def test_tool_choice_any(self):
        tools = [ToolDefinition(name="search", description="", input_schema=ToolInputSchema())]
        request = _make_request(tools=tools, tool_choice=ToolChoiceAny())
        result = build_converse_request(request, _make_model())

        assert result["toolConfig"]["toolChoice"] == {"any": {}}

    def test_tool_choice_unset_keeps_tools_without_choice(self):
        tools = [ToolDefinition(name="search", description="", input_schema=ToolInputSchema())]
        request = _make_request(tools=tools)
        result = build_converse_request(request, _make_model())

        assert "toolConfig" in result
        assert "toolChoice" not in result["toolConfig"]


class TestContentBlockHandling:
    def test_redacted_thinking_dropped(self):
        messages = [
            Message(
                role="assistant",
                content=[
                    RedactedThinkingBlock(data="encrypted"),
                    TextBlock(text="answer"),
                ],
            )
        ]
        result = build_converse_request(_make_request(messages=messages), _make_model())
        content = result["messages"][0]["content"]

        assert content == [{"text": "answer"}]

    def test_server_tool_use_dropped(self):
        messages = [
            Message(
                role="assistant",
                content=[
                    ServerToolUseBlock(id="st_1", name="web_search", input={"q": "x"}),
                    TextBlock(text="done"),
                ],
            )
        ]
        result = build_converse_request(_make_request(messages=messages), _make_model())
        content = result["messages"][0]["content"]

        assert content == [{"text": "done"}]

    def test_tool_result_with_image_content(self):
        messages = [
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="tu_1",
                        content=[
                            TextBlock(text="see image"),
                            ImageBlock(source=ImageSource(media_type="image/png", data="aWdub3Jl")),
                        ],
                    ),
                ],
            )
        ]
        result = build_converse_request(_make_request(messages=messages), _make_model())
        tr_content = result["messages"][0]["content"][0]["toolResult"]["content"]

        assert tr_content[0] == {"text": "see image"}
        assert "image" in tr_content[1]
        assert tr_content[1]["image"]["format"] == "png"
        assert isinstance(tr_content[1]["image"]["source"]["bytes"], bytes)


class TestUsageSchemaParity:
    def test_bedrock_leaves_new_fields_null(self):
        response = {
            "output": {"message": {"content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 3},
        }
        result, _, _ = parse_converse_response(response, "claude-sonnet", uuid.uuid4())
        dumped = result.usage.model_dump()

        # New parity fields exist in the shape but are null for Bedrock.
        assert dumped["cache_creation"] is None
        assert dumped["server_tool_use"] is None
        assert dumped["service_tier"] is None
        # Existing fields still populated.
        assert dumped["input_tokens"] == 10
        assert dumped["cache_read_input_tokens"] == 3

    def test_full_usage_roundtrip(self):
        from ttllm.schemas.anthropic import CacheCreation, ServerToolUsage, Usage

        usage = Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=50,
            cache_creation_input_tokens=30,
            cache_creation=CacheCreation(ephemeral_5m_input_tokens=30),
            server_tool_use=ServerToolUsage(web_search_requests=2),
            service_tier="standard",
        )
        dumped = usage.model_dump()
        assert dumped["cache_creation"]["ephemeral_5m_input_tokens"] == 30
        assert dumped["server_tool_use"]["web_search_requests"] == 2
        assert dumped["service_tier"] == "standard"


def _make_stream_response(events):
    """Build a fake converse_stream response whose 'stream' records pulls."""

    class RecordingStream:
        def __init__(self, items):
            self._items = list(items)
            self.pulled = []

        def __iter__(self):
            return self

        def __next__(self):
            if not self._items:
                raise StopIteration
            item = self._items.pop(0)
            self.pulled.append(item)
            if isinstance(item, Exception):
                raise item
            return item

    return {"stream": RecordingStream(events)}


def _parse_sse(raw: str):
    lines = raw.strip().split("\n")
    event_type = lines[0].replace("event: ", "")
    data = json.loads(lines[1].replace("data: ", ""))
    return event_type, data


class TestStreamConverse:
    @pytest.mark.asyncio
    async def test_token_propagation_via_usage_out(self):
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {
                "inputTokens": 100, "outputTokens": 40,
                "cacheReadInputTokens": 50, "cacheWriteInputTokens": 30,
            }}},
        ]
        model = _make_model()
        usage: dict[str, int] = {}

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = []
            async for ev in stream_converse(_make_request(), model, uuid.uuid4(), usage):
                collected.append(ev)

        assert usage == {
            "input_tokens": 100,
            "output_tokens": 40,
            "cache_read_tokens": 50,
            "cache_write_tokens": 30,
        }
        # message_delta reports the full cumulative usage to the client
        delta = next(_parse_sse(e) for e in collected if e.startswith("event: message_delta"))
        assert delta[1]["usage"]["output_tokens"] == 40
        assert delta[1]["usage"]["input_tokens"] == 100
        assert delta[1]["usage"]["cache_read_input_tokens"] == 50
        assert delta[1]["usage"]["cache_creation_input_tokens"] == 30

    @pytest.mark.asyncio
    async def test_state_populated_through_gateway(self):
        from decimal import Decimal

        from ttllm.core import gateway

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {
                "inputTokens": 100, "outputTokens": 40,
                "cacheReadInputTokens": 50, "cacheWriteInputTokens": 30,
            }}},
        ]
        model = _make_model()
        model.input_cost_per_1k = 0.003
        model.output_cost_per_1k = 0.015
        model.cache_read_cost_per_1k = 0.0003
        model.cache_write_cost_per_1k = 0.00375

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            state, sse_stream = gateway.stream(_make_request(), model, uuid.uuid4())
            async for _ in sse_stream:
                pass

        assert state.input_tokens == 100
        assert state.output_tokens == 40
        assert state.cache_read_tokens == 50
        assert state.cache_write_tokens == 30

        expected = (
            (Decimal("100") / 1000) * Decimal("0.003")
            + (Decimal("40") / 1000) * Decimal("0.015")
            + (Decimal("50") / 1000) * Decimal("0.0003")
            + (Decimal("30") / 1000) * Decimal("0.00375")
        )
        assert state.get_cost() == expected

        # The state rebuilt the full response from the streamed deltas.
        response = state.get_response()
        assert response.content[0].text == "Hi"
        assert response.stop_reason == "end_turn"

        # The metadata blob carries the raw payload and the cost breakdown.
        meta = state.get_metadata()
        assert meta["cost"]["total"] == str(expected)
        assert meta["raw"]["cacheReadInputTokens"] == 50

    @pytest.mark.asyncio
    async def test_incremental_yielding(self):
        """Events are pulled lazily, not drained up front."""
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "a"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "b"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
        ]
        response = _make_stream_response(events)
        model = _make_model()

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = response
            mock_get_client.return_value = mock_client

            gen = stream_converse(_make_request(), model, uuid.uuid4())
            # Pull the first SSE event; the source stream must NOT be fully drained.
            await gen.__anext__()
            pulled_after_first = len(response["stream"].pulled)
            rest = [ev async for ev in gen]

        assert pulled_after_first < len(events)
        assert rest  # remaining events were produced

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        from ttllm.core.bedrock import stream_converse

        model = _make_model()
        usage: dict[str, int] = {}

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response([])
            mock_get_client.return_value = mock_client

            collected = [ev async for ev in stream_converse(_make_request(), model, uuid.uuid4(), usage)]

        types = [_parse_sse(e)[0] for e in collected]
        assert types == ["message_delta", "message_stop"]
        assert usage["input_tokens"] == 0

    @pytest.mark.asyncio
    async def test_stream_exception_mid_iteration(self):
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            RuntimeError("boom"),
        ]
        model = _make_model()

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [ev async for ev in stream_converse(_make_request(), model, uuid.uuid4())]

        types = [_parse_sse(e)[0] for e in collected]
        assert "error" in types

    @pytest.mark.asyncio
    async def test_text_block_opened_without_bedrock_start(self):
        """Bedrock omits contentBlockStart/Stop for text blocks; the gateway must
        still emit content_block_start before the first delta and a matching
        content_block_stop before message_delta, or clients raise
        'Content block not found' and fall back to non-streaming."""
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " world"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
        model = _make_model()

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [ev async for ev in stream_converse(_make_request(), model, uuid.uuid4())]

        types = [_parse_sse(e)[0] for e in collected]
        # A content_block_start for index 0 precedes any delta on it.
        first_delta = types.index("content_block_delta")
        first_start = types.index("content_block_start")
        assert first_start < first_delta
        start_data = _parse_sse(collected[first_start])[1]
        assert start_data["index"] == 0
        assert start_data["content_block"]["type"] == "text"
        # The lazily-opened block is closed before message_delta.
        assert "content_block_stop" in types
        assert types.index("content_block_stop") < types.index("message_delta")

    @pytest.mark.asyncio
    async def test_reasoning_block_opened_without_bedrock_start(self):
        """A reasoning delta with no preceding start opens a thinking block."""
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "reasoningContent": {"text": "thinking..."}
            }}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
        model = _make_model()

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [ev async for ev in stream_converse(_make_request(), model, uuid.uuid4())]

        start = next(_parse_sse(e) for e in collected if e.startswith("event: content_block_start"))
        assert start[1]["content_block"]["type"] == "thinking"

    @pytest.mark.asyncio
    async def test_explicit_tool_use_start_not_double_opened(self):
        """When Bedrock does send contentBlockStart (tool use), the gateway must
        not emit a second, synthetic start for the same index."""
        from ttllm.core.bedrock import stream_converse

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {
                "toolUse": {"toolUseId": "tu_1", "name": "get_weather"}
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '{"city":'}
            }}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "toolUse": {"input": '"NYC"}'}
            }}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
        model = _make_model()

        with patch("ttllm.core.bedrock.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [ev async for ev in stream_converse(_make_request(), model, uuid.uuid4())]

        types = [_parse_sse(e)[0] for e in collected]
        # Exactly one start and one stop for the single tool-use block.
        assert types.count("content_block_start") == 1
        assert types.count("content_block_stop") == 1
        start = next(_parse_sse(e) for e in collected if e.startswith("event: content_block_start"))
        assert start[1]["content_block"]["type"] == "tool_use"
