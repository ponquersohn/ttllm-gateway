"""Pydantic v2 schemas matching the Anthropic Messages API wire format."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag


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


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextBlock] = ""
    is_error: bool = False


def _get_content_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "text")
    if hasattr(v, "type"):
        return v.type
    return "text"


ContentBlock = Annotated[
    Annotated[TextBlock, Tag("text")]
    | Annotated[ImageBlock, Tag("image")]
    | Annotated[ToolUseBlock, Tag("tool_use")]
    | Annotated[ToolResultBlock, Tag("tool_result")],
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


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"


class ToolChoiceAny(BaseModel):
    type: Literal["any"] = "any"


class ToolChoiceTool(BaseModel):
    type: Literal["tool"] = "tool"
    name: str


ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool


# --- Messages ---


class Message(BaseModel):
    role: Literal["user", "assistant"]
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
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    """Anthropic Messages API response format."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
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


class StreamEventContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | InputJsonDelta


class StreamEventContentBlockStop(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class StreamDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    stop_reason: str | None = None
    stop_sequence: str | None = None


class StreamUsage(BaseModel):
    output_tokens: int


class StreamEventMessageDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: StreamDelta
    usage: StreamUsage


class StreamEventMessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class StreamEventPing(BaseModel):
    type: Literal["ping"] = "ping"
