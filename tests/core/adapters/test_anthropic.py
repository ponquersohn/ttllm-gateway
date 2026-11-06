"""Tests for the Anthropic wire format <-> internal representation adapter."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ttllm.core.adapters.anthropic import _part_to_block, request_to_internal, result_to_response
from ttllm.core.errors import UnsupportedContentError
from ttllm.core.model import Citation, InternalResult, InternalUsage, TextPart, ThinkingPart, ToolCallPart
from ttllm.schemas.anthropic import (
    DocumentBlock,
    DocumentSource,
    ImageBlock,
    ImageSource,
    Message,
    MessagesRequest,
    ServerToolDefinition,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceNone,
    ToolChoiceTool,
    ToolDefinition,
    ToolInputSchema,
    ToolResultBlock,
    ToolUseBlock,
    WebSearchToolResultBlock,
)


def _make_model(**overrides):
    model = MagicMock()
    model.provider_model_id = "anthropic.claude-sonnet-4-20250514-v1:0"
    model.name = "claude-sonnet"
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


class TestServerToolConversion:
    """The adapter no longer rejects server-side tool content -- whether a provider can
    proxy it is a provider-layer decision (see core/providers/bedrock/converse.py,
    core/providers/langchain/translation.py), so
    the adapter just converts it faithfully like everything else."""

    def test_server_tool_definition_converted_not_rejected(self):
        tools = [
            ToolDefinition(name="search", description="", input_schema=ToolInputSchema()),
            ServerToolDefinition(type="web_search_20250305", name="web_search"),
        ]
        internal = request_to_internal(_make_request(tools=tools), _make_model())
        assert len(internal.tools) == 1
        assert internal.tools[0].name == "search"
        assert len(internal.server_tools) == 1
        assert internal.server_tools[0].type == "web_search_20250305"
        assert internal.server_tools[0].name == "web_search"

    def test_server_tool_definition_extra_fields_captured(self):
        tools = [ServerToolDefinition(type="web_search_20250305", name="web_search", max_uses=3)]
        internal = request_to_internal(_make_request(tools=tools), _make_model())
        assert internal.server_tools[0].config == {"max_uses": 3}

    def test_server_tool_use_block_in_history_converted(self):
        request = _make_request(
            messages=[
                Message(role="user", content="search for cats"),
                Message(
                    role="assistant",
                    content=[ServerToolUseBlock(id="st_1", name="web_search", input={"q": "cats"})],
                ),
            ]
        )
        internal = request_to_internal(request, _make_model())
        part = internal.messages[1].content[0]
        assert part.type == "server_tool_call"
        assert part.name == "web_search"
        assert part.input == {"q": "cats"}

    def test_web_search_tool_result_block_in_history_converted(self):
        request = _make_request(
            messages=[
                Message(role="user", content="hi"),
                Message(role="assistant", content=[WebSearchToolResultBlock(tool_use_id="st_1", content=[])]),
            ]
        )
        internal = request_to_internal(request, _make_model())
        part = internal.messages[1].content[0]
        assert part.type == "server_tool_result"
        assert part.tool_call_id == "st_1"

    def test_no_server_tools_gives_empty_list(self):
        tools = [ToolDefinition(name="search", description="", input_schema=ToolInputSchema())]
        internal = request_to_internal(_make_request(tools=tools), _make_model())
        assert internal.server_tools == []


class TestRequestToInternal:
    def test_basic_text_message(self):
        internal = request_to_internal(_make_request(), _make_model())
        assert internal.provider_model_id == "anthropic.claude-sonnet-4-20250514-v1:0"
        assert len(internal.messages) == 1
        assert internal.messages[0].role == "user"
        assert internal.messages[0].content == [TextPart(text="Hello")]

    def test_system_string(self):
        internal = request_to_internal(_make_request(system="Be helpful."), _make_model())
        assert internal.system == [TextPart(text="Be helpful.")]

    def test_system_as_text_blocks(self):
        internal = request_to_internal(
            _make_request(system=[TextBlock(text="Part 1"), TextBlock(text="Part 2")]), _make_model()
        )
        assert internal.system == [TextPart(text="Part 1"), TextPart(text="Part 2")]

    def test_system_cache_control(self):
        internal = request_to_internal(
            _make_request(system=[TextBlock(text="prefix", cache_control={"type": "ephemeral"})]),
            _make_model(),
        )
        assert internal.system == [TextPart(text="prefix", cache_control=True)]

    def test_system_cache_boundary_preserved_across_blocks(self):
        """The bug this guards against: a cached top-level block followed by an
        uncached one must stay two separate parts, not get merged into a single
        span -- merging would make Bedrock cache (and re-cache, on every request)
        whatever dynamic content follows the cache_control marker."""
        internal = request_to_internal(
            _make_request(
                system=[
                    TextBlock(text="static instructions", cache_control={"type": "ephemeral"}),
                    TextBlock(text="dynamic per-request tail"),
                ]
            ),
            _make_model(),
        )
        assert internal.system == [
            TextPart(text="static instructions", cache_control=True),
            TextPart(text="dynamic per-request tail", cache_control=False),
        ]

    def test_mid_conversation_system_lifted(self):
        request = _make_request(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="system", content="Terse mode enabled."),
                Message(role="user", content="Continue"),
            ]
        )
        internal = request_to_internal(request, _make_model())
        assert internal.system == [TextPart(text="Terse mode enabled.")]
        assert [m.role for m in internal.messages] == ["user", "user"]

    def test_mid_conversation_system_appended_after_top_level(self):
        request = _make_request(
            system="Base prompt.",
            messages=[
                Message(role="user", content="Hi"),
                Message(role="system", content="Switch to JSON output."),
            ],
        )
        internal = request_to_internal(request, _make_model())
        assert internal.system == [
            TextPart(text="Base prompt."),
            TextPart(text="Switch to JSON output."),
        ]

    def test_mid_conversation_system_stays_uncached_after_cached_top_level(self):
        """A cached top-level system prompt followed by a later mid-conversation
        system message (typically per-turn dynamic content) must not be folded into
        the same cached span as the top-level block."""
        request = _make_request(
            system=[TextBlock(text="Base prompt.", cache_control={"type": "ephemeral"})],
            messages=[
                Message(role="user", content="Hi"),
                Message(role="system", content="Switch to JSON output."),
            ],
        )
        internal = request_to_internal(request, _make_model())
        assert internal.system == [
            TextPart(text="Base prompt.", cache_control=True),
            TextPart(text="Switch to JSON output.", cache_control=False),
        ]

    def test_system_message_block_content_flattened(self):
        request = _make_request(
            messages=[
                Message(role="user", content="Hi"),
                Message(role="system", content=[TextBlock(text="Line A"), TextBlock(text="Line B")]),
            ]
        )
        internal = request_to_internal(request, _make_model())
        assert internal.system == [TextPart(text="Line A\nLine B")]

    def test_tool_result_str_content_normalized_to_list(self):
        request = _make_request(
            messages=[
                Message(role="user", content=[ToolResultBlock(tool_use_id="tu_1", content="Result text")]),
            ]
        )
        internal = request_to_internal(request, _make_model())
        result_part = internal.messages[0].content[0]
        assert result_part.type == "tool_result"
        assert result_part.content == [TextPart(text="Result text")]

    def test_tool_result_image_content_preserved(self):
        # This is the root-cause fix for the translator crash: the internal model
        # never has a str|list ambiguity, every consumer sees a uniform list.
        request = _make_request(
            messages=[
                Message(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="tu_1",
                            content=[
                                TextBlock(text="see image"),
                                ImageBlock(source=ImageSource(media_type="image/png", data="aWdub3Jl")),
                            ],
                        )
                    ],
                )
            ]
        )
        internal = request_to_internal(request, _make_model())
        result_part = internal.messages[0].content[0]
        assert len(result_part.content) == 2
        assert result_part.content[0].type == "text"
        assert result_part.content[1].type == "image"

    def test_tool_result_document_content_preserved(self):
        # Claude Code's Read tool returns PDFs as a document block inside tool_result.
        request = MessagesRequest.model_validate({
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": [
                        {"type": "text", "text": "PDF file read"},
                        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                        "data": "aWdub3Jl"}},
                    ],
                }],
            }],
        })
        internal = request_to_internal(request, _make_model())
        result_part = internal.messages[0].content[0]
        assert [p.type for p in result_part.content] == ["text", "document"]
        assert result_part.content[1].media_type == "application/pdf"
        assert result_part.content[1].data == "aWdub3Jl"

    def test_document_block_conversion(self):
        request = _make_request(
            messages=[
                Message(
                    role="user",
                    content=[DocumentBlock(source=DocumentSource(media_type="application/pdf", data="AA=="), title="a.pdf")],
                )
            ]
        )
        internal = request_to_internal(request, _make_model())
        part = internal.messages[0].content[0]
        assert part.type == "document"
        assert part.title == "a.pdf"

    def test_thinking_and_tool_use_conversion(self):
        request = _make_request(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ThinkingBlock(thinking="hmm", signature="sig"),
                        ToolUseBlock(id="tu_1", name="search", input={"q": "x"}),
                    ],
                )
            ]
        )
        internal = request_to_internal(request, _make_model())
        parts = internal.messages[0].content
        assert isinstance(parts[0], ThinkingPart)
        assert isinstance(parts[1], ToolCallPart)

    @pytest.mark.parametrize(
        "tool_choice,expected_mode,expected_name",
        [
            (ToolChoiceAuto(), "auto", None),
            (ToolChoiceAny(), "any", None),
            (ToolChoiceTool(name="search"), "tool", "search"),
            (ToolChoiceNone(), "none", None),
            (None, None, None),
        ],
    )
    def test_tool_choice_mapping(self, tool_choice, expected_mode, expected_name):
        request = _make_request(
            tools=[ToolDefinition(name="search", description="", input_schema=ToolInputSchema())],
            tool_choice=tool_choice,
        )
        internal = request_to_internal(request, _make_model())
        if expected_mode is None:
            assert internal.tool_choice is None
        else:
            assert internal.tool_choice.mode == expected_mode
            assert internal.tool_choice.tool_name == expected_name

    def test_tools_carry_cache_control(self):
        tools = [
            ToolDefinition(
                name="search",
                description="Search",
                input_schema=ToolInputSchema(properties={"q": {"type": "string"}}, required=["q"]),
                cache_control={"type": "ephemeral"},
            )
        ]
        internal = request_to_internal(_make_request(tools=tools), _make_model())
        assert internal.tools[0].name == "search"
        assert internal.tools[0].cache_control is True
        assert internal.tools[0].input_schema["properties"] == {"q": {"type": "string"}}

    def test_thinking_config(self):
        request = _make_request(thinking={"type": "enabled", "budget_tokens": 2000})
        internal = request_to_internal(request, _make_model())
        assert internal.thinking.enabled is True
        assert internal.thinking.type == "enabled"
        assert internal.thinking.budget_tokens == 2000

    def test_no_thinking(self):
        internal = request_to_internal(_make_request(), _make_model())
        assert internal.thinking is None

    def test_thinking_disabled(self):
        request = _make_request(thinking={"type": "disabled"})
        internal = request_to_internal(request, _make_model())
        assert internal.thinking.enabled is False

    def test_thinking_adaptive(self):
        request = _make_request(thinking={"type": "adaptive"})
        internal = request_to_internal(request, _make_model())
        assert internal.thinking.enabled is True
        assert internal.thinking.type == "adaptive"
        assert internal.thinking.budget_tokens is None

    def test_thinking_enabled_without_budget_tokens_raises(self):
        request = _make_request(thinking={"type": "enabled"})
        with pytest.raises(UnsupportedContentError):
            request_to_internal(request, _make_model())


class TestResultToResponse:
    def test_basic_round_trip(self):
        result = InternalResult(
            content=[TextPart(text="Hi!")],
            stop_reason="end_turn",
            usage=InternalUsage(input_tokens=10, output_tokens=5),
        )
        request_id = uuid.uuid4()
        response = result_to_response(result, "claude-sonnet", request_id)

        assert response.id == f"msg_{request_id.hex[:24]}"
        assert response.model == "claude-sonnet"
        assert response.stop_reason == "end_turn"
        assert response.content[0].text == "Hi!"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    def test_cache_tokens_mapped(self):
        result = InternalResult(
            content=[TextPart(text="cached")],
            stop_reason="end_turn",
            usage=InternalUsage(input_tokens=100, output_tokens=20, cache_read_tokens=50, cache_write_tokens=30),
        )
        response = result_to_response(result, "claude-sonnet", uuid.uuid4())
        assert response.usage.cache_read_input_tokens == 50
        assert response.usage.cache_creation_input_tokens == 30

    def test_empty_content_gets_empty_text_block(self):
        result = InternalResult(content=[], stop_reason="end_turn", usage=InternalUsage(input_tokens=1, output_tokens=0))
        response = result_to_response(result, "claude-sonnet", uuid.uuid4())
        assert len(response.content) == 1
        assert response.content[0].text == ""

    def test_tool_call_round_trip(self):
        result = InternalResult(
            content=[ToolCallPart(id="tu_1", name="search", input={"q": "x"})],
            stop_reason="tool_use",
            usage=InternalUsage(input_tokens=1, output_tokens=1),
        )
        response = result_to_response(result, "claude-sonnet", uuid.uuid4())
        assert response.content[0].type == "tool_use"
        assert response.content[0].id == "tu_1"


class TestSafeguards:
    _SAFEGUARDS = [{"type": "dangerous_tool_use", "classifier_context": {"v": 1}}]

    def test_request_safeguards_carried_verbatim(self):
        internal = request_to_internal(_make_request(safeguards=self._SAFEGUARDS), _make_model())
        assert internal.safeguards == self._SAFEGUARDS

    def test_request_without_safeguards(self):
        assert request_to_internal(_make_request(), _make_model()).safeguards == []

    def test_wire_request_keeps_safeguards_field(self):
        wire = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "safeguards": self._SAFEGUARDS}
        assert MessagesRequest.model_validate(wire).safeguards == self._SAFEGUARDS

    def test_results_on_response(self):
        results = [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}}]
        result = InternalResult(
            content=[TextPart(text="ok")],
            stop_reason="end_turn",
            usage=InternalUsage(input_tokens=1, output_tokens=1),
            safeguard_results=results,
        )
        assert result_to_response(result, "claude-sonnet", uuid.uuid4()).safeguard_results == results


def _req(content, **extra) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": content}], **extra}
    )


def _first_part(content, **extra):
    return request_to_internal(_req(content, **extra), _make_model()).messages[0].content[0]


class TestCompatibility:
    """Valid Anthropic input must never fail schema validation (a 422 kills the client's
    conversation); anything the gateway can't handle becomes a readable 400 instead."""

    def test_plain_text_document_source(self):
        part = _first_part([{"type": "document", "title": "Notes", "context": "meeting notes",
                             "citations": {"enabled": True},
                             "source": {"type": "text", "media_type": "text/plain", "data": "hello"}}])
        assert (part.source_kind, part.text, part.context, part.citations) == ("text", "hello", "meeting notes", True)

    def test_content_document_source(self):
        part = _first_part([{"type": "document", "source": {"type": "content", "content": [
            {"type": "text", "text": "chunk 1"}, {"type": "text", "text": "chunk 2"}]}}])
        assert part.source_kind == "content"
        assert [p.text for p in part.content] == ["chunk 1", "chunk 2"]

    @pytest.mark.parametrize("block", [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
        {"type": "image", "source": {"type": "file", "file_id": "file_1"}},
        {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}},
        {"type": "document", "source": {"type": "file", "file_id": "file_1"}},
    ])
    def test_url_and_file_sources_are_400_not_422(self, block):
        with pytest.raises(UnsupportedContentError, match="inline as base64"):
            _first_part([block])

    def test_unknown_block_is_400_not_422(self):
        with pytest.raises(UnsupportedContentError, match="'container_upload'"):
            _first_part([{"type": "container_upload", "file_id": "f"}])

    def test_unknown_block_inside_tool_result_is_400(self):
        with pytest.raises(UnsupportedContentError, match="'brand_new'"):
            _first_part([{"type": "tool_result", "tool_use_id": "t", "content": [{"type": "brand_new"}]}])

    def test_search_result_top_level_and_in_tool_result(self):
        sr = {"type": "search_result", "source": "https://kb/1", "title": "KB 1",
              "content": [{"type": "text", "text": "fact"}], "citations": {"enabled": True}}
        top = _first_part([sr])
        assert (top.type, top.source, top.title, top.citations) == ("search_result", "https://kb/1", "KB 1", True)
        nested = _first_part([{"type": "tool_result", "tool_use_id": "t", "content": [sr]}])
        assert nested.content[0].type == "search_result"

    def test_generic_server_tool_blocks_round_trip(self):
        call = _first_part([{"type": "mcp_tool_use", "id": "m1", "name": "x", "server_name": "srv", "input": {}}])
        assert (call.call_type, call.extra) == ("mcp_tool_use", {"server_name": "srv"})
        result = _first_part([{"type": "web_fetch_tool_result", "tool_use_id": "s1",
                               "content": {"type": "web_fetch_result", "url": "u"}}])
        assert result.result_type == "web_fetch_tool_result"
        block = _part_to_block(result).model_dump()
        assert block["type"] == "web_fetch_tool_result"
        assert block["content"] == {"type": "web_fetch_result", "url": "u"}

    def test_tool_schema_kept_whole(self):
        schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}},
                  "$defs": {"A": {"type": "string"}}, "additionalProperties": False}
        internal = request_to_internal(_req("hi", tools=[{"name": "t", "input_schema": schema}]), _make_model())
        dumped = internal.tools[0].input_schema
        assert dumped["$defs"] == {"A": {"type": "string"}}
        assert dumped["additionalProperties"] is False

    def test_unknown_top_level_params_logged_not_rejected(self, caplog):
        with caplog.at_level("WARNING"):
            request_to_internal(_req("hi", container="c1", context_management={}), _make_model())
        assert "container" in caplog.text and "context_management" in caplog.text

    def test_citations_emitted_on_output(self):
        part = TextPart(text="x", citations=[
            Citation(kind="char", cited_text="q", document_index=0, document_title="D", start=1, end=4),
            Citation(kind="search_result", cited_text="f", search_result_index=2, source="s", title="T", start=0, end=1),
        ])
        block = _part_to_block(part).model_dump()
        assert block["citations"] == [
            {"type": "char_location", "cited_text": "q", "document_index": 0, "document_title": "D",
             "start_char_index": 1, "end_char_index": 4},
            {"type": "search_result_location", "cited_text": "f", "search_result_index": 2, "source": "s",
             "title": "T", "start_block_index": 0, "end_block_index": 1},
        ]

    def test_input_citations_parsed(self):
        msg = {"type": "text", "text": "x", "citations": [
            {"type": "page_location", "cited_text": "q", "document_index": 1, "document_title": None,
             "start_page_number": 2, "end_page_number": 3}]}
        part = _first_part([msg])
        assert part.citations[0].kind == "page" and (part.citations[0].start, part.citations[0].end) == (2, 3)
