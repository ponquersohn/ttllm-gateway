"""Tests for direct Bedrock Converse integration."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from ttllm.core.providers.bedrock.converse import build_converse_request, parse_converse_response
from ttllm.core.errors import ServerToolError
from ttllm.core.model import (
    ChunkKind,
    DocumentPart,
    ImagePart,
    InternalMessage,
    InternalRequest,
    RedactedThinkingPart,
    ServerToolCallPart,
    ServerToolResultPart,
    ServerToolSpec,
    TextPart,
    ThinkingConfig,
    ThinkingPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
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


def _make_request(**kwargs) -> InternalRequest:
    defaults = {
        "provider_model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 1024,
        "messages": [InternalMessage(role="user", content=[TextPart(text="Hello")])],
    }
    defaults.update(kwargs)
    return InternalRequest(**defaults)


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

    def test_system_message(self):
        request = _make_request(system="Be helpful.")
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["system"] == [{"text": "Be helpful."}]

    def test_system_cache_control(self):
        request = _make_request(system="Big shared prefix.", system_cache_control=True)
        result = build_converse_request(request, _make_model())

        assert result["system"] == [
            {"text": "Big shared prefix."},
            {"cachePoint": {"type": "default"}},
        ]

    def test_no_system_omits_key(self):
        request = _make_request()
        result = build_converse_request(request, _make_model())
        assert "system" not in result

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
        request = _make_request(thinking=ThinkingConfig(enabled=True, budget_tokens=5000))
        model = _make_model()
        result = build_converse_request(request, model)

        assert result["additionalModelRequestFields"]["thinking"] == {
            "type": "enabled",
            "budget_tokens": 5000,
        }

    def test_thinking_disabled_omitted(self):
        request = _make_request(thinking=ThinkingConfig(enabled=False))
        result = build_converse_request(request, _make_model())
        assert "additionalModelRequestFields" not in result

    def test_tool_definitions(self):
        tools = [
            ToolSpec(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            )
        ]
        request = _make_request(tools=tools, tool_choice=ToolChoice(mode="auto"))
        model = _make_model()
        result = build_converse_request(request, model)

        assert "toolConfig" in result
        tool_specs = result["toolConfig"]["tools"]
        assert len(tool_specs) == 1
        assert tool_specs[0]["toolSpec"]["name"] == "search"
        assert tool_specs[0]["toolSpec"]["inputSchema"]["json"]["properties"] == {"query": {"type": "string"}}
        assert result["toolConfig"]["toolChoice"] == {"auto": {}}

    def test_image_part_conversion(self):
        messages = [
            InternalMessage(
                role="user",
                content=[
                    TextPart(text="What is this?"),
                    ImagePart(media_type="image/png", data="aWdub3Jl"),
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

    def test_tool_call_part_conversion(self):
        messages = [
            InternalMessage(
                role="assistant",
                content=[ToolCallPart(id="tu_123", name="search", input={"q": "test"})],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["toolUse"]["toolUseId"] == "tu_123"
        assert content[0]["toolUse"]["name"] == "search"
        assert content[0]["toolUse"]["input"] == {"q": "test"}

    def test_tool_result_part_conversion(self):
        messages = [
            InternalMessage(
                role="user",
                content=[
                    ToolResultPart(
                        tool_call_id="tu_123",
                        content=[TextPart(text="Result text")],
                        is_error=False,
                    ),
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
            InternalMessage(
                role="user",
                content=[
                    ToolResultPart(tool_call_id="tu_456", content=[TextPart(text="Failed")], is_error=True),
                ],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["toolResult"]["status"] == "error"

    def test_tool_result_with_image_content(self):
        messages = [
            InternalMessage(
                role="user",
                content=[
                    ToolResultPart(
                        tool_call_id="tu_1",
                        content=[
                            TextPart(text="see image"),
                            ImagePart(media_type="image/png", data="aWdub3Jl"),
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

    def test_thinking_part_conversion(self):
        messages = [
            InternalMessage(
                role="assistant",
                content=[
                    ThinkingPart(text="Let me think about this...", signature="sig_abc"),
                    TextPart(text="Here is my answer."),
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

    def test_document_part_conversion(self):
        messages = [
            InternalMessage(
                role="user",
                content=[DocumentPart(media_type="application/pdf", data="SlZCRVI=", title="report.pdf")],
            )
        ]
        request = _make_request(messages=messages)
        model = _make_model()
        result = build_converse_request(request, model)

        content = result["messages"][0]["content"]
        assert content[0]["document"]["format"] == "pdf"
        assert content[0]["document"]["name"] == "report.pdf"

    def test_redacted_thinking_dropped(self):
        # Bedrock has no equivalent for redacted-thinking content -- Anthropic itself
        # defines it as opaque, so this is the one deliberate silent drop, not a
        # translation gap.
        messages = [
            InternalMessage(
                role="assistant",
                content=[RedactedThinkingPart(data="encrypted"), TextPart(text="answer")],
            )
        ]
        result = build_converse_request(_make_request(messages=messages), _make_model())
        content = result["messages"][0]["content"]

        assert content == [{"text": "answer"}]


class TestServerToolsRejected:
    """Whether a provider can proxy server-side tools is a provider decision -- the
    adapter converts them faithfully, and Bedrock (which can't proxy them) rejects here,
    at translation time, not upstream."""

    def test_server_tool_definition_raises(self):
        request = _make_request(server_tools=[ServerToolSpec(type="web_search_20250305", name="web_search")])
        with pytest.raises(ServerToolError):
            build_converse_request(request, _make_model())

    def test_server_tool_call_in_history_raises(self):
        messages = [
            InternalMessage(
                role="assistant",
                content=[ServerToolCallPart(id="st_1", name="web_search", input={"q": "cats"})],
            )
        ]
        with pytest.raises(ServerToolError):
            build_converse_request(_make_request(messages=messages), _make_model())

    def test_server_tool_result_in_history_raises(self):
        messages = [
            InternalMessage(
                role="user",
                content=[ServerToolResultPart(tool_call_id="st_1", content=[])],
            )
        ]
        with pytest.raises(ServerToolError):
            build_converse_request(_make_request(messages=messages), _make_model())


class TestParseConverseResponse:
    def test_basic_text_response(self):
        response = {
            "output": {"message": {"content": [{"text": "Hello!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }
        result = parse_converse_response(response)

        assert result.stop_reason == "end_turn"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hello!"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.cache_read_tokens == 0
        assert result.usage.cache_write_tokens == 0

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
        result = parse_converse_response(response)

        assert result.stop_reason == "tool_use"
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "tool_call"
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
        result = parse_converse_response(response)

        assert len(result.content) == 2
        assert result.content[0].type == "thinking"
        assert result.content[0].text == "Thinking deeply..."
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
        result = parse_converse_response(response)

        # cache_read is reported separately, NOT folded into billed input_tokens.
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 20
        assert result.usage.cache_read_tokens == 50
        assert result.usage.cache_write_tokens == 30

    def test_max_tokens_stop_reason(self):
        response = {
            "output": {"message": {"content": [{"text": "truncated..."}]}},
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 10, "outputTokens": 4096},
        }
        result = parse_converse_response(response)

        assert result.stop_reason == "max_tokens"

    def test_empty_content_gets_empty_text_block(self):
        response = {
            "output": {"message": {"content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 5, "outputTokens": 0},
        }
        result = parse_converse_response(response)

        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == ""


class TestInvokeConverse:
    @pytest.mark.asyncio
    async def test_invoke_calls_boto3(self):
        from ttllm.core.providers.bedrock.converse import invoke_converse

        model = _make_model()
        request = _make_request()

        mock_response = {
            "output": {"message": {"content": [{"text": "Hi there!"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 8, "outputTokens": 4},
        }

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = await invoke_converse(request, model)

        assert result.content[0].text == "Hi there!"
        assert result.usage.input_tokens == 8
        mock_client.converse.assert_called_once()


class TestGetBoto3Client:
    def test_endpoint_url_passed_to_client(self):
        from ttllm.core.providers.bedrock import converse as bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={
            "region": "us-east-1",
            "endpoint_url": "http://fake-bedrock:9099",
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
        })

        with patch("ttllm.core.providers.bedrock.converse.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        mock_session.return_value.client.assert_called_once()
        args, kwargs = mock_session.return_value.client.call_args
        assert args[0] == "bedrock-runtime"
        assert kwargs["endpoint_url"] == "http://fake-bedrock:9099"

    def test_no_endpoint_url_omits_kwarg(self):
        from ttllm.core.providers.bedrock import converse as bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={"region": "us-east-1"})

        with patch("ttllm.core.providers.bedrock.converse.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        assert "endpoint_url" not in kwargs

    def test_default_timeouts_applied(self):
        from ttllm.core.providers.bedrock import converse as bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={"region": "us-east-1"})

        with patch("ttllm.core.providers.bedrock.converse.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        boto_config = kwargs["config"]
        assert boto_config.read_timeout == bedrock._DEFAULT_READ_TIMEOUT
        assert boto_config.connect_timeout == bedrock._DEFAULT_CONNECT_TIMEOUT
        assert boto_config.retries == {"mode": "standard", "max_attempts": bedrock._DEFAULT_MAX_ATTEMPTS}

    def test_config_json_timeout_overrides(self):
        from ttllm.core.providers.bedrock import converse as bedrock

        bedrock._CLIENT_CACHE.clear()
        model = _make_model(config_json={
            "region": "us-east-1",
            "read_timeout": 600,
            "connect_timeout": 5,
            "retry_max_attempts": 1,
        })

        with patch("ttllm.core.providers.bedrock.converse.boto3.Session") as mock_session:
            bedrock.get_boto3_client(model)

        _, kwargs = mock_session.return_value.client.call_args
        boto_config = kwargs["config"]
        assert boto_config.read_timeout == 600
        assert boto_config.connect_timeout == 5
        assert boto_config.retries == {"mode": "standard", "max_attempts": 1}

    def test_distinct_timeouts_get_distinct_cached_clients(self):
        from ttllm.core.providers.bedrock import converse as bedrock

        bedrock._CLIENT_CACHE.clear()
        base = {"region": "us-east-1"}
        model_default = _make_model(config_json=base)
        model_tuned = _make_model(config_json={**base, "read_timeout": 600})

        with patch("ttllm.core.providers.bedrock.converse.boto3.Session") as mock_session:
            # A fresh mock per client() call, so identity reflects cache behavior.
            mock_session.return_value.client.side_effect = lambda *a, **k: MagicMock()
            client_a = bedrock.get_boto3_client(model_default)
            client_b = bedrock.get_boto3_client(model_tuned)
            client_a_again = bedrock.get_boto3_client(model_default)

        assert client_a is not client_b
        assert client_a is client_a_again


class TestToolChoice:
    def test_tool_choice_none_omits_tool_config(self):
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        request = _make_request(tools=tools, tool_choice=ToolChoice(mode="none"))
        result = build_converse_request(request, _make_model())

        assert "toolConfig" not in result

    def test_tool_choice_any(self):
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        request = _make_request(tools=tools, tool_choice=ToolChoice(mode="any"))
        result = build_converse_request(request, _make_model())

        assert result["toolConfig"]["toolChoice"] == {"any": {}}

    def test_tool_choice_specific_tool(self):
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        request = _make_request(tools=tools, tool_choice=ToolChoice(mode="tool", tool_name="search"))
        result = build_converse_request(request, _make_model())

        assert result["toolConfig"]["toolChoice"] == {"tool": {"name": "search"}}

    def test_tool_choice_unset_keeps_tools_without_choice(self):
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        request = _make_request(tools=tools)
        result = build_converse_request(request, _make_model())

        assert "toolConfig" in result
        assert "toolChoice" not in result["toolConfig"]


class TestUsageSchemaParity:
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


class TestStreamConverseChunks:
    @pytest.mark.asyncio
    async def test_token_propagation_and_message_stop(self):
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

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

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        stop = next(c for c in collected if c.kind == ChunkKind.MESSAGE_STOP)
        assert stop.stop_reason == "end_turn"
        assert stop.usage.input_tokens == 100
        assert stop.usage.output_tokens == 40
        assert stop.usage.cache_read_tokens == 50
        assert stop.usage.cache_write_tokens == 30

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

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            state, chunks = gateway.stream(_make_request(), model, uuid.uuid4())
            async for _ in chunks:
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
        result = state.get_response()
        assert result.content[0].text == "Hi"
        assert result.stop_reason == "end_turn"

        meta = state.get_metadata()
        assert meta["cost"]["total"] == str(expected)
        assert meta["raw"]["cacheReadInputTokens"] == 50

    @pytest.mark.asyncio
    async def test_incremental_yielding(self):
        """Events are pulled lazily, not drained up front."""
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "a"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "b"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 2}}},
        ]
        response = _make_stream_response(events)
        model = _make_model()

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = response
            mock_get_client.return_value = mock_client

            gen = stream_converse_chunks(_make_request(), model, uuid.uuid4())
            await gen.__anext__()
            pulled_after_first = len(response["stream"].pulled)
            rest = [c async for c in gen]

        assert pulled_after_first < len(events)
        assert rest

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

        model = _make_model()

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response([])
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        assert [c.kind for c in collected] == [ChunkKind.MESSAGE_STOP]
        assert collected[0].usage.input_tokens == 0

    @pytest.mark.asyncio
    async def test_stream_exception_mid_iteration(self):
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

        events = [
            {"messageStart": {"role": "assistant"}},
            RuntimeError("boom"),
        ]
        model = _make_model()

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        assert any(c.kind == ChunkKind.ERROR for c in collected)

    @pytest.mark.asyncio
    async def test_text_block_opened_without_bedrock_start(self):
        """Bedrock omits contentBlockStart/Stop for text blocks; the chunk producer must
        still emit a BLOCK_START before the first delta and a matching BLOCK_STOP before
        message_stop, or the SSE encoder downstream would violate the wire protocol."""
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": " world"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
        model = _make_model()

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        kinds = [c.kind for c in collected]
        first_delta = kinds.index(ChunkKind.TEXT_DELTA)
        first_start = kinds.index(ChunkKind.BLOCK_START)
        assert first_start < first_delta
        assert collected[first_start].index == 0
        assert collected[first_start].part.type == "text"
        assert ChunkKind.BLOCK_STOP in kinds
        assert kinds.index(ChunkKind.BLOCK_STOP) < kinds.index(ChunkKind.MESSAGE_STOP)

    @pytest.mark.asyncio
    async def test_reasoning_block_opened_without_bedrock_start(self):
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
                "reasoningContent": {"text": "thinking..."}
            }}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 5, "outputTokens": 2}}},
        ]
        model = _make_model()

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        start = next(c for c in collected if c.kind == ChunkKind.BLOCK_START)
        assert start.part.type == "thinking"

    @pytest.mark.asyncio
    async def test_explicit_tool_use_start_not_double_opened(self):
        """When Bedrock does send contentBlockStart (tool use), the producer must not
        emit a second, synthetic start for the same index."""
        from ttllm.core.providers.bedrock.converse import stream_converse_chunks

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

        with patch("ttllm.core.providers.bedrock.converse.get_boto3_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.converse_stream.return_value = _make_stream_response(events)
            mock_get_client.return_value = mock_client

            collected = [c async for c in stream_converse_chunks(_make_request(), model, uuid.uuid4())]

        kinds = [c.kind for c in collected]
        assert kinds.count(ChunkKind.BLOCK_START) == 1
        assert kinds.count(ChunkKind.BLOCK_STOP) == 1
        start = next(c for c in collected if c.kind == ChunkKind.BLOCK_START)
        assert start.part.type == "tool_call"


class TestCachePoint:
    def test_tool_cache_control_emits_cache_point(self):
        tools = [
            ToolSpec(
                name="search",
                description="Search",
                input_schema={"type": "object", "properties": {}},
                cache_control=True,
            ),
        ]
        request = _make_request(tools=tools)
        result = build_converse_request(request, _make_model())

        tool_specs = result["toolConfig"]["tools"]
        assert len(tool_specs) == 2
        assert tool_specs[0]["toolSpec"]["name"] == "search"
        assert tool_specs[1] == {"cachePoint": {"type": "default"}}

    def test_message_text_part_cache_control_emits_cache_point(self):
        messages = [
            InternalMessage(
                role="user",
                content=[
                    TextPart(text="Cache me.", cache_control=True),
                    TextPart(text="But not me."),
                ],
            )
        ]
        request = _make_request(messages=messages)
        result = build_converse_request(request, _make_model())

        content = result["messages"][0]["content"]
        assert content == [
            {"text": "Cache me."},
            {"cachePoint": {"type": "default"}},
            {"text": "But not me."},
        ]

    def test_no_cache_control_emits_no_cache_point(self):
        tools = [ToolSpec(name="search", input_schema={"type": "object", "properties": {}})]
        request = _make_request(
            system="prefix",
            tools=tools,
            messages=[InternalMessage(role="user", content=[TextPart(text="hello")])],
        )
        result = build_converse_request(request, _make_model())

        def _has_cache_point(arr):
            return any("cachePoint" in el for el in arr)

        assert not _has_cache_point(result["system"])
        assert not _has_cache_point(result["toolConfig"]["tools"])
        assert not _has_cache_point(result["messages"][0]["content"])
