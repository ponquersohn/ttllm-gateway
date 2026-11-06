"""Pydantic v2 schemas matching the Anthropic Messages API wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# --- Content blocks ---


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str
    # Accepted on input (replayed assistant turns) and emitted on output when the model
    # cites a document/search result. Entries are Anthropic citation-location dicts.
    citations: list[dict[str, Any]] | None = None
    cache_control: dict[str, Any] | None = None


class ImageSource(BaseModel):
    """Base64 source (images and binary documents)."""

    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class UrlSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["url"] = "url"
    url: str


class FileSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["file"] = "file"
    file_id: str


def _get_source_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "base64")
    return getattr(v, "type", "base64")


AnyImageSource = Annotated[
    Annotated[ImageSource, Tag("base64")]
    | Annotated[UrlSource, Tag("url")]
    | Annotated[FileSource, Tag("file")],
    Discriminator(_get_source_type),
]


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: AnyImageSource
    cache_control: dict[str, Any] | None = None


class DocumentSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class PlainTextSource(BaseModel):
    type: Literal["text"] = "text"
    media_type: str = "text/plain"
    data: str


class ContentSource(BaseModel):
    type: Literal["content"] = "content"
    content: str | list[TextBlock | ImageBlock]


AnyDocumentSource = Annotated[
    Annotated[DocumentSource, Tag("base64")]
    | Annotated[PlainTextSource, Tag("text")]
    | Annotated[ContentSource, Tag("content")]
    | Annotated[UrlSource, Tag("url")]
    | Annotated[FileSource, Tag("file")],
    Discriminator(_get_source_type),
]


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    source: AnyDocumentSource
    title: str | None = None
    context: str | None = None
    citations: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None


class SearchResultBlock(BaseModel):
    type: Literal["search_result"] = "search_result"
    source: str
    title: str
    content: list[TextBlock]
    citations: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    cache_control: dict[str, Any] | None = None


class ServerToolUseBlock(BaseModel):
    # ``mcp_tool_use`` (MCP connector) has the same shape plus ``server_name``.
    model_config = ConfigDict(extra="allow")

    type: Literal["server_tool_use", "mcp_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: dict[str, Any]


class WebSearchToolResultBlock(BaseModel):
    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: list[Any] = []


# Server-tool result block types other than web_search. Their payloads are each
# server tool's own contract, so they're kept opaque (``extra="allow"``).
SERVER_TOOL_RESULT_TYPES = frozenset({
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "tool_search_tool_result",
    "mcp_tool_result",
})


class ServerToolResultBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    tool_use_id: str
    content: Any = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[ToolResultContentBlock] = ""
    is_error: bool = False
    cache_control: dict[str, Any] | None = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class UnknownBlock(BaseModel):
    """Any content block type this gateway doesn't model. Parsing it (instead of
    failing validation with a 422) lets the input adapter reject it with a readable
    400 naming the type."""

    model_config = ConfigDict(extra="allow")

    type: str


_KNOWN_CONTENT_TYPES = frozenset({
    "text", "image", "document", "search_result", "tool_use", "server_tool_use",
    "web_search_tool_result", "tool_result", "thinking", "redacted_thinking",
})


def _get_content_type(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type", "text")
    else:
        t = getattr(v, "type", "text")
    if t in _KNOWN_CONTENT_TYPES:
        return t
    if t == "mcp_tool_use":
        return "server_tool_use"
    if t in SERVER_TOOL_RESULT_TYPES:
        return "server_tool_result"
    return "unknown"


def _get_tool_result_content_type(v: Any) -> str:
    t = _get_content_type(v)
    return t if t in ("text", "image", "document", "search_result") else "unknown"


ToolResultContentBlock = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ImageBlock, Tag("image")]
    | Annotated[DocumentBlock, Tag("document")]
    | Annotated[SearchResultBlock, Tag("search_result")]
    | Annotated[UnknownBlock, Tag("unknown")],
    Discriminator(_get_tool_result_content_type),
]

ToolResultBlock.model_rebuild()


ContentBlock = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ImageBlock, Tag("image")]
    | Annotated[DocumentBlock, Tag("document")]
    | Annotated[SearchResultBlock, Tag("search_result")]
    | Annotated[ToolUseBlock, Tag("tool_use")]
    | Annotated[ServerToolUseBlock, Tag("server_tool_use")]
    | Annotated[WebSearchToolResultBlock, Tag("web_search_tool_result")]
    | Annotated[ServerToolResultBlock, Tag("server_tool_result")]
    | Annotated[ToolResultBlock, Tag("tool_result")]
    | Annotated[ThinkingBlock, Tag("thinking")]
    | Annotated[RedactedThinkingBlock, Tag("redacted_thinking")]
    | Annotated[UnknownBlock, Tag("unknown")],
    Discriminator(_get_content_type),
]


# --- Tool definitions ---


class ToolInputSchema(BaseModel):
    # Keep the whole JSON Schema ($defs, additionalProperties, anyOf, description, ...),
    # not just properties/required -- MCP tools in particular rely on the rest.
    model_config = ConfigDict(extra="allow")

    type: Literal["object"] = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: ToolInputSchema
    cache_control: dict[str, Any] | None = None


class ServerToolDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"
    disable_parallel_tool_use: bool | None = None


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str
    disable_parallel_tool_use: bool | None = None


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool | ToolChoiceNone


# --- Messages ---


class Message(BaseModel):
    # "system" is accepted mid-array for Anthropic's mid-conversation system
    # messages (beta mid-conversation-system-2026-04-07). Bedrock Converse has no
    # inline system role, so the translator lifts these into the top-level system.
    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]


class MessagesRequest(BaseModel):
    """Anthropic Messages API request format."""

    # Unknown top-level params (container, mcp_servers, context_management, ...) are
    # kept in ``model_extra`` so the adapter can log what it isn't forwarding, rather
    # than having them vanish without a trace.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    max_tokens: int
    system: str | list[TextBlock] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    metadata: dict[str, Any] | None = None
    tools: list[ToolDefinition | ServerToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    thinking: dict[str, Any] | None = None
    service_tier: str | None = None
    # Server-side safety checks (beta dangerous-tool-use-2026-09-03), sent by Claude
    # Code in auto mode. Opaque: the entries are forwarded as-is to providers that
    # support them.
    safeguards: list[dict[str, Any]] | None = None


class CacheCreation(BaseModel):
    ephemeral_5m_input_tokens: int | None = None
    ephemeral_1h_input_tokens: int | None = None


class ServerToolUsage(BaseModel):
    web_search_requests: int | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation: CacheCreation | None = None
    server_tool_use: ServerToolUsage | None = None
    service_tier: str | None = None


class StopDetails(BaseModel):
    type: str
    value: str | None = None


class MessagesResponse(BaseModel):
    """Anthropic Messages API response format."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    stop_details: StopDetails | None = None
    usage: Usage
    # Present only when the request carried ``safeguards`` and the provider ran them;
    # left out of the wire response otherwise (see ``result_to_response``).
    safeguard_results: list[dict[str, Any]] | None = None


# --- Streaming events ---


class StreamEventMessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessagesResponse


class StreamEventContentBlockStart(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class SignatureDelta(BaseModel):
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


class StreamEventContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | InputJsonDelta | ThinkingDelta | SignatureDelta


class StreamEventContentBlockStop(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class StreamDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    stop_reason: str | None = None
    stop_sequence: str | None = None


class StreamUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


class StreamEventMessageDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: StreamDelta
    usage: StreamUsage


class StreamEventMessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class StreamEventPing(BaseModel):
    type: Literal["ping"] = "ping"


# --- Models API ---


class AnthropicModelObject(BaseModel):
    type: Literal["model"] = "model"
    id: str
    display_name: str
    created_at: str


class AnthropicModelListResponse(BaseModel):
    data: list[AnthropicModelObject]
    has_more: bool
    first_id: str | None = None
    last_id: str | None = None
