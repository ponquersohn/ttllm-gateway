"""Anthropic wire format <-> internal representation.

The only place that knows Anthropic-specific quirks: mid-conversation ``role="system"``
messages, ``cache_control`` markers, and the exact ``tool_choice``/content-block shapes.
Everything downstream (``core/gateway.py``, both providers) only ever sees
``ttllm.core.model`` types.
"""

from __future__ import annotations

import uuid
from typing import Any

from ttllm.core.errors import UnsupportedContentError
from ttllm.core.model import (
    DocumentPart,
    ImagePart,
    InternalMessage,
    InternalRequest,
    InternalResult,
    Part,
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
from ttllm.schemas.anthropic import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingBlock,
    ServerToolDefinition,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolChoiceAny,
    ToolChoiceAuto,
    ToolChoiceNone,
    ToolChoiceTool,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    WebSearchToolResultBlock,
    DocumentSource,
)


def _tool_result_content_to_parts(content: str | list[TextBlock | ImageBlock]) -> list[TextPart | ImagePart]:
    if isinstance(content, str):
        return [TextPart(text=content)] if content else []
    parts: list[TextPart | ImagePart] = []
    for b in content:
        if isinstance(b, TextBlock):
            parts.append(TextPart(text=b.text, cache_control=bool(b.cache_control)))
        elif isinstance(b, ImageBlock):
            parts.append(ImagePart(media_type=b.source.media_type, data=b.source.data))
    return parts


def _block_to_part(block: ContentBlock) -> Part:
    if isinstance(block, TextBlock):
        return TextPart(text=block.text, cache_control=bool(block.cache_control))
    if isinstance(block, ImageBlock):
        return ImagePart(media_type=block.source.media_type, data=block.source.data)
    if isinstance(block, DocumentBlock):
        return DocumentPart(media_type=block.source.media_type, data=block.source.data, title=block.title)
    if isinstance(block, ToolUseBlock):
        return ToolCallPart(id=block.id, name=block.name, input=block.input)
    if isinstance(block, ToolResultBlock):
        return ToolResultPart(
            tool_call_id=block.tool_use_id,
            content=_tool_result_content_to_parts(block.content),
            is_error=block.is_error,
        )
    if isinstance(block, ThinkingBlock):
        return ThinkingPart(text=block.thinking, signature=block.signature)
    if isinstance(block, RedactedThinkingBlock):
        return RedactedThinkingPart(data=block.data)
    if isinstance(block, ServerToolUseBlock):
        return ServerToolCallPart(id=block.id, name=block.name, input=block.input)
    if isinstance(block, WebSearchToolResultBlock):
        return ServerToolResultPart(tool_call_id=block.tool_use_id, content=list(block.content))
    raise AssertionError(f"Unhandled content block type {type(block).__name__!r}")


def _content_to_parts(content: str | list[ContentBlock]) -> list[Part]:
    if isinstance(content, str):
        return [TextPart(text=content)] if content else []
    return [_block_to_part(b) for b in content]


def _message_system_text(msg: Message) -> str:
    """Flatten a mid-conversation system-role message's content into plain text.
    Non-text blocks don't occur on a system turn in practice; drop them defensively."""
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))


def _flatten_system(request: MessagesRequest) -> tuple[str | None, bool]:
    """Returns (flattened system text, system_cache_control). Ordering matches today's
    Bedrock behavior: top-level ``system`` text first, then each inline
    ``role="system"`` message's text, in conversation order."""
    lines: list[str] = []
    cache_control = False
    if request.system:
        if isinstance(request.system, str):
            lines.append(request.system)
        else:
            for block in request.system:
                lines.append(block.text)
                if block.cache_control:
                    cache_control = True
    for msg in request.messages:
        if msg.role == "system":
            text = _message_system_text(msg)
            if text:
                lines.append(text)
    return ("\n".join(lines) if lines else None), cache_control


def _tool_choice_to_internal(tool_choice: Any) -> ToolChoice | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, ToolChoiceAuto):
        return ToolChoice(mode="auto")
    if isinstance(tool_choice, ToolChoiceAny):
        return ToolChoice(mode="any")
    if isinstance(tool_choice, ToolChoiceTool):
        return ToolChoice(mode="tool", tool_name=tool_choice.name)
    if isinstance(tool_choice, ToolChoiceNone):
        return ToolChoice(mode="none")
    return None


def _client_tools_to_internal(tools: list[ToolDefinition | ServerToolDefinition] | None) -> list[ToolSpec]:
    if not tools:
        return []
    return [
        ToolSpec(
            name=t.name,
            description=t.description,
            input_schema=t.input_schema.model_dump(),
            cache_control=bool(t.cache_control),
        )
        for t in tools
        if isinstance(t, ToolDefinition)
    ]


def _server_tools_to_internal(tools: list[ToolDefinition | ServerToolDefinition] | None) -> list[ServerToolSpec]:
    if not tools:
        return []
    specs = []
    for t in tools:
        if not isinstance(t, ServerToolDefinition):
            continue
        extra = t.model_dump(exclude={"type", "name"})
        specs.append(ServerToolSpec(type=t.type, name=t.name, config=extra))
    return specs


def _thinking_to_internal(thinking: dict[str, Any] | None) -> ThinkingConfig | None:
    if not thinking:
        return None
    thinking_type = thinking.get("type", "enabled")
    if thinking_type == "disabled":
        return ThinkingConfig(enabled=False)
    if thinking_type == "adaptive":
        return ThinkingConfig(enabled=True, type="adaptive")
    budget_tokens = thinking.get("budget_tokens")
    if not isinstance(budget_tokens, int):
        raise UnsupportedContentError(
            "thinking.budget_tokens is required and must be an integer when thinking.type is 'enabled'"
        )
    return ThinkingConfig(enabled=True, type="enabled", budget_tokens=budget_tokens)


def request_to_internal(request: MessagesRequest, llm_model: Any) -> InternalRequest:
    """Convert an Anthropic ``MessagesRequest`` into the internal representation.

    Faithfully converts every Anthropic content type, including server-side tool
    definitions/content -- whether a given provider can actually handle those is a
    provider-layer decision (``ServerToolError``, raised from provider translation code),
    not something this adapter should presume.
    """
    system, system_cache_control = _flatten_system(request)

    messages = [
        InternalMessage(role=msg.role, content=_content_to_parts(msg.content))
        for msg in request.messages
        if msg.role != "system"
    ]

    return InternalRequest(
        provider_model_id=llm_model.provider_model_id,
        messages=messages,
        system=system,
        system_cache_control=system_cache_control,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences or [],
        tools=_client_tools_to_internal(request.tools),
        server_tools=_server_tools_to_internal(request.tools),
        tool_choice=_tool_choice_to_internal(request.tool_choice),
        thinking=_thinking_to_internal(request.thinking),
    )


def _part_to_block(part: Part) -> ContentBlock:
    if isinstance(part, TextPart):
        return TextBlock(text=part.text)
    if isinstance(part, ImagePart):
        return ImageBlock(source=ImageSource(media_type=part.media_type, data=part.data))
    if isinstance(part, DocumentPart):
        return DocumentBlock(
            source=DocumentSource(media_type=part.media_type, data=part.data), title=part.title
        )
    if isinstance(part, ToolCallPart):
        return ToolUseBlock(id=part.id, name=part.name, input=part.input)
    if isinstance(part, ThinkingPart):
        return ThinkingBlock(thinking=part.text, signature=part.signature)
    if isinstance(part, RedactedThinkingPart):
        return RedactedThinkingBlock(data=part.data)
    if isinstance(part, ServerToolCallPart):
        return ServerToolUseBlock(id=part.id, name=part.name, input=part.input)
    if isinstance(part, ServerToolResultPart):
        return WebSearchToolResultBlock(tool_use_id=part.tool_call_id, content=part.content)
    raise AssertionError(f"Unexpected part {type(part).__name__} in a provider result")


def result_to_response(result: InternalResult, model_name: str, request_id: uuid.UUID) -> MessagesResponse:
    """Convert an ``InternalResult`` into the Anthropic ``MessagesResponse`` wire shape."""
    content_blocks = [_part_to_block(p) for p in result.content]
    if not content_blocks:
        content_blocks.append(TextBlock(text=""))

    usage = Usage(
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cache_read_input_tokens=result.usage.cache_read_tokens or None,
        cache_creation_input_tokens=result.usage.cache_write_tokens or None,
    )

    return MessagesResponse(
        id=f"msg_{request_id.hex[:24]}",
        content=content_blocks,
        model=model_name,
        stop_reason=result.stop_reason,
        usage=usage,
    )
