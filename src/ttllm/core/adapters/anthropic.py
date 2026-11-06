"""Anthropic wire format <-> internal representation.

The only place that knows Anthropic-specific quirks: mid-conversation ``role="system"``
messages, ``cache_control`` markers, and the exact ``tool_choice``/content-block shapes.
Everything downstream (``core/gateway.py``, both providers) only ever sees
``ttllm.core.model`` types.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ttllm.core.errors import UnsupportedContentError
from ttllm.core.model import (
    Citation,
    DocumentPart,
    ImagePart,
    InternalMessage,
    InternalRequest,
    InternalResult,
    Part,
    RedactedThinkingPart,
    SearchResultPart,
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
    ContentSource,
    DocumentBlock,
    ImageBlock,
    ImageSource,
    Message,
    MessagesRequest,
    MessagesResponse,
    PlainTextSource,
    RedactedThinkingBlock,
    SearchResultBlock,
    ServerToolDefinition,
    ServerToolResultBlock,
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
    UnknownBlock,
    Usage,
    WebSearchToolResultBlock,
    DocumentSource,
)

logger = logging.getLogger(__name__)


# --- Citations (wire <-> internal) ---

_WIRE_CITATION_KIND = {
    "char_location": "char",
    "page_location": "page",
    "content_block_location": "chunk",
    "search_result_location": "search_result",
    "web_search_result_location": "web",
}
_INDEX_FIELDS = {
    "char": ("start_char_index", "end_char_index"),
    "page": ("start_page_number", "end_page_number"),
    "chunk": ("start_block_index", "end_block_index"),
    "search_result": ("start_block_index", "end_block_index"),
}


def _citation_from_wire(c: dict[str, Any]) -> Citation | None:
    kind = _WIRE_CITATION_KIND.get(c.get("type", ""))
    if kind is None:
        return None
    start_key, end_key = _INDEX_FIELDS.get(kind, ("", ""))
    return Citation(
        kind=kind,
        cited_text=c.get("cited_text", ""),
        document_index=c.get("document_index"),
        document_title=c.get("document_title"),
        search_result_index=c.get("search_result_index"),
        start=c.get(start_key),
        end=c.get(end_key),
        source=c.get("source"),
        title=c.get("title"),
        url=c.get("url"),
    )


def citation_to_wire(c: Citation) -> dict[str, Any]:
    """Internal ``Citation`` -> Anthropic citation-location dict (shared with the SSE encoder)."""
    if c.kind == "web":
        return {"type": "web_search_result_location", "cited_text": c.cited_text,
                "url": c.url or "", "title": c.title, "encrypted_index": ""}
    start_key, end_key = _INDEX_FIELDS[c.kind]
    out: dict[str, Any] = {"cited_text": c.cited_text}
    if c.kind == "search_result":
        out.update(type="search_result_location", search_result_index=c.search_result_index or 0,
                   source=c.source or "", title=c.title)
    else:
        out.update(type={"char": "char_location", "page": "page_location",
                         "chunk": "content_block_location"}[c.kind],
                   document_index=c.document_index or 0, document_title=c.document_title)
    out[start_key] = c.start or 0
    out[end_key] = c.end or 0
    return out


# --- Content blocks -> parts ---


def _unsupported_source(kind: str, source_type: str) -> UnsupportedContentError:
    if source_type == "file":
        return UnsupportedContentError(
            f"{kind.capitalize()} sources of type 'file' (Files API) are not supported by this "
            f"gateway; send the {kind} inline as base64."
        )
    return UnsupportedContentError(
        f"{kind.capitalize()} sources of type {source_type!r} are not supported by this gateway; "
        f"send the {kind} inline as base64."
    )


def _text_to_part(b: TextBlock) -> TextPart:
    citations = None
    if b.citations:
        citations = [c for c in (_citation_from_wire(d) for d in b.citations) if c is not None] or None
    return TextPart(text=b.text, citations=citations, cache_control=bool(b.cache_control))


def _image_to_part(b: ImageBlock) -> ImagePart:
    if not isinstance(b.source, ImageSource):
        raise _unsupported_source("image", b.source.type)
    return ImagePart(media_type=b.source.media_type, data=b.source.data, cache_control=bool(b.cache_control))


def _document_to_part(b: DocumentBlock) -> DocumentPart:
    common: dict[str, Any] = dict(
        title=b.title,
        context=b.context,
        citations=bool((b.citations or {}).get("enabled")),
        cache_control=bool(b.cache_control),
    )
    src = b.source
    if isinstance(src, DocumentSource):
        return DocumentPart(source_kind="base64", media_type=src.media_type, data=src.data, **common)
    if isinstance(src, PlainTextSource):
        return DocumentPart(source_kind="text", media_type=src.media_type, text=src.data, **common)
    if isinstance(src, ContentSource):
        if isinstance(src.content, str):
            inner: list[TextPart | ImagePart] = [TextPart(text=src.content)]
        else:
            inner = [_text_to_part(c) if isinstance(c, TextBlock) else _image_to_part(c) for c in src.content]
        return DocumentPart(source_kind="content", media_type="text/plain", content=inner, **common)
    raise _unsupported_source("document", src.type)


def _search_result_to_part(b: SearchResultBlock) -> SearchResultPart:
    return SearchResultPart(
        source=b.source,
        title=b.title,
        content=[TextPart(text=t.text) for t in b.content],
        citations=bool((b.citations or {}).get("enabled")),
        # Converse can't put a cachePoint inside searchResult.content; promote inner markers.
        cache_control=bool(b.cache_control) or any(t.cache_control for t in b.content),
    )


def _unknown_block(b: UnknownBlock) -> UnsupportedContentError:
    return UnsupportedContentError(f"Content block type {b.type!r} is not supported by this gateway.")


def _tool_result_content_to_parts(
    content: str | list[Any],
) -> list[TextPart | ImagePart | DocumentPart | SearchResultPart]:
    if isinstance(content, str):
        return [TextPart(text=content)] if content else []
    parts: list[TextPart | ImagePart | DocumentPart | SearchResultPart] = []
    for b in content:
        if isinstance(b, TextBlock):
            parts.append(_text_to_part(b))
        elif isinstance(b, ImageBlock):
            parts.append(_image_to_part(b))
        elif isinstance(b, DocumentBlock):
            parts.append(_document_to_part(b))
        elif isinstance(b, SearchResultBlock):
            parts.append(_search_result_to_part(b))
        elif isinstance(b, UnknownBlock):
            raise _unknown_block(b)
    return parts


def _block_to_part(block: ContentBlock) -> Part:
    if isinstance(block, TextBlock):
        return _text_to_part(block)
    if isinstance(block, ImageBlock):
        return _image_to_part(block)
    if isinstance(block, DocumentBlock):
        return _document_to_part(block)
    if isinstance(block, SearchResultBlock):
        return _search_result_to_part(block)
    if isinstance(block, ToolUseBlock):
        return ToolCallPart(id=block.id, name=block.name, input=block.input,
                            cache_control=bool(block.cache_control))
    if isinstance(block, ToolResultBlock):
        inner = _tool_result_content_to_parts(block.content)
        return ToolResultPart(
            tool_call_id=block.tool_use_id,
            content=inner,
            is_error=block.is_error,
            # A marker on the tool_result itself, or on its last inner text block, both mean
            # "cache the prefix through this tool result". Converse cannot place a cachePoint
            # inside toolResult.content, so promote it to the sibling position.
            cache_control=bool(block.cache_control) or any(p.cache_control for p in inner),
        )
    if isinstance(block, ThinkingBlock):
        return ThinkingPart(text=block.thinking, signature=block.signature)
    if isinstance(block, RedactedThinkingBlock):
        return RedactedThinkingPart(data=block.data)
    if isinstance(block, ServerToolUseBlock):
        return ServerToolCallPart(id=block.id, name=block.name, input=block.input,
                                  call_type=block.type, extra=dict(block.model_extra or {}))
    if isinstance(block, WebSearchToolResultBlock):
        return ServerToolResultPart(tool_call_id=block.tool_use_id, content=list(block.content))
    if isinstance(block, ServerToolResultBlock):
        return ServerToolResultPart(tool_call_id=block.tool_use_id, content=block.content,
                                    result_type=block.type, extra=dict(block.model_extra or {}))
    if isinstance(block, UnknownBlock):
        raise _unknown_block(block)
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


def _flatten_system(request: MessagesRequest) -> list[TextPart] | None:
    """Returns the system content as an ordered list of parts: top-level ``system``
    blocks first, then each inline ``role="system"`` message's text, in conversation
    order -- matching today's Bedrock ordering. Each part keeps its own
    ``cache_control``, so a cache breakpoint on one block never silently swallows
    whatever comes after it (a trailing uncached top-level block, or a later
    mid-conversation system message) into the same cached span."""
    parts: list[TextPart] = []
    if request.system:
        if isinstance(request.system, str):
            parts.append(TextPart(text=request.system))
        else:
            for block in request.system:
                parts.append(TextPart(text=block.text, cache_control=bool(block.cache_control)))
    for msg in request.messages:
        if msg.role == "system":
            text = _message_system_text(msg)
            if text:
                # A marker on any block within this system message means "cache the
                # prefix through this message" -- same promotion rule used for
                # ToolResultPart above, since Converse has no way to place a
                # cachePoint mid-message.
                inner_cache_control = (
                    isinstance(msg.content, list)
                    and any(isinstance(b, TextBlock) and b.cache_control for b in msg.content)
                )
                parts.append(TextPart(text=text, cache_control=bool(inner_cache_control)))
    return parts or None


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
    if request.model_extra:
        logger.warning(
            "Ignoring unsupported top-level request params: %s", ", ".join(sorted(request.model_extra))
        )

    system = _flatten_system(request)

    messages = [
        InternalMessage(role=msg.role, content=_content_to_parts(msg.content))
        for msg in request.messages
        if msg.role != "system"
    ]

    return InternalRequest(
        provider_model_id=llm_model.provider_model_id,
        messages=messages,
        system=system,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop_sequences=request.stop_sequences or [],
        tools=_client_tools_to_internal(request.tools),
        server_tools=_server_tools_to_internal(request.tools),
        tool_choice=_tool_choice_to_internal(request.tool_choice),
        thinking=_thinking_to_internal(request.thinking),
        safeguards=request.safeguards or [],
    )


def _part_to_block(part: Part) -> ContentBlock:
    if isinstance(part, TextPart):
        citations = [citation_to_wire(c) for c in part.citations] if part.citations else None
        return TextBlock(text=part.text, citations=citations)
    if isinstance(part, ImagePart):
        return ImageBlock(source=ImageSource(media_type=part.media_type, data=part.data))
    if isinstance(part, DocumentPart):
        if part.source_kind == "text":
            source: Any = PlainTextSource(media_type=part.media_type, data=part.text or "")
        elif part.source_kind == "content":
            source = ContentSource(content=[_part_to_block(p) for p in part.content or []])
        else:
            source = DocumentSource(media_type=part.media_type, data=part.data)
        return DocumentBlock(
            source=source, title=part.title, context=part.context,
            citations={"enabled": True} if part.citations else None,
        )
    if isinstance(part, SearchResultPart):
        return SearchResultBlock(
            source=part.source, title=part.title, content=[TextBlock(text=t.text) for t in part.content],
            citations={"enabled": True} if part.citations else None,
        )
    if isinstance(part, ToolCallPart):
        return ToolUseBlock(id=part.id, name=part.name, input=part.input)
    if isinstance(part, ThinkingPart):
        return ThinkingBlock(thinking=part.text, signature=part.signature)
    if isinstance(part, RedactedThinkingPart):
        return RedactedThinkingBlock(data=part.data)
    if isinstance(part, ServerToolCallPart):
        return ServerToolUseBlock(type=part.call_type, id=part.id, name=part.name, input=part.input,
                                  **part.extra)
    if isinstance(part, ServerToolResultPart):
        if part.result_type == "web_search_tool_result":
            return WebSearchToolResultBlock(tool_use_id=part.tool_call_id, content=part.content)
        return ServerToolResultBlock(type=part.result_type, tool_use_id=part.tool_call_id,
                                     content=part.content, **part.extra)
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
        safeguard_results=result.safeguard_results,
    )
