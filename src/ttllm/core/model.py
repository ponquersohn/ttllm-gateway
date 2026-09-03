"""Provider-agnostic, API-agnostic internal representation.

This is the pivot type between wire formats: an input adapter (``core/adapters/``)
converts an external wire request into an ``InternalRequest``, a provider
(``core/providers/bedrock/``, ``core/providers/langchain/``) converts that into its own wire
format and converts the response back into an ``InternalResult``/``InternalChunk`` stream,
and the input adapter converts that back into the external wire response.

Whether a given provider can actually send a given piece of content is a *provider*
decision, not something this module or an input adapter should presume -- so every content
type an input API can express gets a ``Part``/spec here, even ones no provider currently
supports (e.g. server-side tools). Each provider's translation code decides, per part, to
represent it, emulate it with something non-misleading, or reject it (``UnsupportedContentError``
for content it can't interpret at all, ``ServerToolError`` specifically for server-side
tools it can't proxy). See ``docs/content-mapping.md`` for the full policy.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Tag


# --- Content parts ---


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str
    cache_control: bool = False


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: str
    cache_control: bool = False


class DocumentPart(BaseModel):
    type: Literal["document"] = "document"
    media_type: str
    data: str
    title: str | None = None
    cache_control: bool = False


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any]
    cache_control: bool = False


class ToolResultPart(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: list[TextPart | ImagePart]
    is_error: bool = False
    cache_control: bool = False


class ThinkingPart(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str
    signature: str
    cache_control: bool = False


class RedactedThinkingPart(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str
    cache_control: bool = False


class ServerToolCallPart(BaseModel):
    """A server-side tool invocation (e.g. Anthropic's built-in web_search). No
    provider in this gateway can proxy these today -- see ``docs/content-mapping.md`` --
    but that's a provider-layer decision, so it still gets a first-class representation
    here rather than being excluded from the model entirely."""

    type: Literal["server_tool_call"] = "server_tool_call"
    id: str
    name: str
    input: dict[str, Any]
    cache_control: bool = False


class ServerToolResultPart(BaseModel):
    type: Literal["server_tool_result"] = "server_tool_result"
    tool_call_id: str
    content: list[Any] = []
    cache_control: bool = False


def _get_part_type(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "text")
    return getattr(v, "type", "text")


Part = Annotated[
    Annotated[TextPart, Tag("text")]
    | Annotated[ImagePart, Tag("image")]
    | Annotated[DocumentPart, Tag("document")]
    | Annotated[ToolCallPart, Tag("tool_call")]
    | Annotated[ToolResultPart, Tag("tool_result")]
    | Annotated[ThinkingPart, Tag("thinking")]
    | Annotated[RedactedThinkingPart, Tag("redacted_thinking")]
    | Annotated[ServerToolCallPart, Tag("server_tool_call")]
    | Annotated[ServerToolResultPart, Tag("server_tool_result")],
    Discriminator(_get_part_type),
]


# --- Request ---


class InternalMessage(BaseModel):
    # Anthropic's mid-conversation "system" role (and its top-level `system` field) is
    # flattened away entirely by the input adapter -- it never reaches this type.
    role: Literal["user", "assistant"]
    content: list[Part]


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any]
    cache_control: bool = False


class ServerToolSpec(BaseModel):
    """A server-side tool declaration (e.g. Anthropic's ``web_search_20250305``). Shape
    is a passthrough -- ``type``/``name`` plus whatever extra config the wire format
    attaches -- since each input API's server tools are its own arbitrary contract, not
    something to normalize into a shared schema."""

    type: str
    name: str
    config: dict[str, Any] = {}


class ToolChoice(BaseModel):
    mode: Literal["auto", "any", "none", "tool"]
    tool_name: str | None = None


class ThinkingConfig(BaseModel):
    enabled: bool = False
    # "adaptive" thinking (no fixed budget) vs "enabled" (requires budget_tokens).
    type: Literal["enabled", "adaptive"] = "enabled"
    budget_tokens: int | None = None


class InternalRequest(BaseModel):
    provider_model_id: str
    messages: list[InternalMessage]
    system: str | None = None
    system_cache_control: bool = False
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = []
    tools: list[ToolSpec] = []
    server_tools: list[ServerToolSpec] = []
    tool_choice: ToolChoice | None = None
    thinking: ThinkingConfig | None = None


# --- Result (non-streaming) ---


class InternalUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class InternalResult(BaseModel):
    """A completed exchange, in internal form.

    No ``id``/``model`` fields -- those are wire-envelope concerns, synthesized only by
    an input adapter (e.g. ``core/adapters/anthropic.py::result_to_response``).
    """

    content: list[Part]
    stop_reason: Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    usage: InternalUsage


# --- Streaming ---


class ChunkKind(str, Enum):
    BLOCK_START = "block_start"
    TEXT_DELTA = "text_delta"
    TOOL_ARGS_DELTA = "tool_args_delta"
    THINKING_DELTA = "thinking_delta"
    SIGNATURE_DELTA = "signature_delta"
    BLOCK_STOP = "block_stop"
    MESSAGE_STOP = "message_stop"
    ERROR = "error"


class InternalChunk(BaseModel):
    kind: ChunkKind
    index: int = 0
    part: Part | None = None
    text: str | None = None
    partial_json: str | None = None
    signature: str | None = None
    stop_reason: str | None = None
    usage: InternalUsage | None = None
    error_message: str | None = None
