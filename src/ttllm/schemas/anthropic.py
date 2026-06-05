"""Pydantic v2 schemas matching the Anthropic Messages API wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


# --- Content blocks ---


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource


class DocumentSource(BaseModel):
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ServerToolUseBlock(BaseModel):
    type: Literal["server_tool_use"] = "server_tool_use"
    id: str
    name: str
    input: dict[str, Any]


class WebSearchToolResultBlock(BaseModel):
    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: list[Any] = []


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock] = ""
    is_error: bool = False


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


def _get_content_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "text")
    if hasattr(v, "type"):
        return v.type
    return "text"


ContentBlock = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ImageBlock, Tag("image")]
    | Annotated[DocumentBlock, Tag("document")]
    | Annotated[ToolUseBlock, Tag("tool_use")]
    | Annotated[ServerToolUseBlock, Tag("server_tool_use")]
    | Annotated[WebSearchToolResultBlock, Tag("web_search_tool_result")]
    | Annotated[ToolResultBlock, Tag("tool_result")]
    | Annotated[ThinkingBlock, Tag("thinking")]
    | Annotated[RedactedThinkingBlock, Tag("redacted_thinking")],
    Discriminator(_get_content_type),
]


# --- Tool definitions ---


class ToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, Any] = {}
    required: list[str] = []


class ToolDefinition(BaseModel):
    name: str
    description: str = ""
    input_schema: ToolInputSchema


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
