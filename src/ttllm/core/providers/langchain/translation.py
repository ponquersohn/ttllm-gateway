"""Translate between the internal representation and LangChain messages.

No framework dependencies beyond langchain-core (LangChain messages are this provider's
own wire format, same as boto3 dicts are Bedrock's). Carries zero dependency on the
Anthropic wire schema -- see ``ttllm.core.adapters.anthropic`` for that half of the
translation.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from ttllm.core.errors import ServerToolError, UnsupportedContentError
from ttllm.core.model import (
    DocumentPart,
    ImagePart,
    InternalRequest,
    InternalResult,
    InternalUsage,
    Part,
    RedactedThinkingPart,
    ServerToolCallPart,
    ServerToolResultPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
)


def _convert_parts_to_langchain(parts: list[Part]) -> list[dict]:
    """Convert internal parts (from a user-turn message) to LangChain multimodal
    content-part dicts."""
    result: list[dict] = []
    for part in parts:
        if isinstance(part, TextPart):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            result.append({
                "type": "image_url",
                "image_url": {"url": f"data:{part.media_type};base64,{part.data}"},
            })
        elif isinstance(part, DocumentPart):
            raise UnsupportedContentError(
                "Document content is not supported by the OpenAI-compatible provider."
            )
        elif isinstance(part, ToolCallPart):
            # Only reachable for malformed input (a tool call in a non-assistant turn --
            # valid Anthropic conversations never put one here). Best-effort text stub,
            # not a real feature gap.
            result.append({"type": "text", "text": f"[tool_use: {part.name}({part.input})]"})
        elif isinstance(part, (ServerToolCallPart, ServerToolResultPart)):
            raise ServerToolError(
                "Server-side tools cannot be proxied through the OpenAI-compatible provider. "
                "Remove server tool content and handle it client-side."
            )
        # ThinkingPart/RedactedThinkingPart/ToolResultPart don't occur in user-turn
        # content in a valid conversation; nothing meaningful to do if they somehow do.
    return result


def _tool_result_content_to_langchain(content: list[TextPart | ImagePart]) -> str | list[dict]:
    """Convert a ToolResultPart's content to LangChain ToolMessage content.

    Text-only content (the common case) stays a plain string, matching today's shape.
    Content containing an image is emulated as a multimodal content-part list --
    ToolMessage.content accepts the same list-of-parts shape HumanMessage.content does --
    instead of crashing or dropping the image.
    """
    parts: list[dict] = []
    for p in content:
        if isinstance(p, TextPart):
            parts.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{p.media_type};base64,{p.data}"},
            })
    if not parts:
        return ""
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def to_langchain_messages(request: InternalRequest) -> list[BaseMessage]:
    """Convert an ``InternalRequest`` into a list of LangChain messages."""
    if request.server_tools:
        raise ServerToolError(
            "Server-side tools cannot be proxied through the OpenAI-compatible provider. "
            "Remove server tool definitions and handle them client-side."
        )

    msgs: list[BaseMessage] = []

    if request.system:
        msgs.append(SystemMessage(content=request.system))

    for msg in request.messages:
        if msg.role == "user":
            tool_results = [p for p in msg.content if isinstance(p, ToolResultPart)]
            if tool_results:
                for tr in tool_results:
                    msgs.append(ToolMessage(
                        content=_tool_result_content_to_langchain(tr.content),
                        tool_call_id=tr.tool_call_id,
                    ))
                non_tool = [p for p in msg.content if not isinstance(p, ToolResultPart)]
                if non_tool:
                    msgs.append(HumanMessage(content=_convert_parts_to_langchain(non_tool)))
            else:
                msgs.append(HumanMessage(content=_convert_parts_to_langchain(msg.content)))

        elif msg.role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in msg.content:
                if isinstance(part, ToolCallPart):
                    tool_calls.append({"name": part.name, "args": part.input, "id": part.id})
                elif isinstance(part, TextPart):
                    text_parts.append(part.text)
                elif isinstance(part, ThinkingPart):
                    # Emulate: fold the reasoning text into a normal text part instead
                    # of dropping it -- no native "thinking" concept on this provider.
                    text_parts.append(part.text)
                elif isinstance(part, RedactedThinkingPart):
                    # Silent drop: the payload is encrypted, not even Anthropic can
                    # read it, so there's no content actually being lost here.
                    continue
                elif isinstance(part, DocumentPart):
                    raise UnsupportedContentError(
                        "Document content is not supported by the OpenAI-compatible provider."
                    )
                elif isinstance(part, (ServerToolCallPart, ServerToolResultPart)):
                    raise ServerToolError(
                        "Server-side tools cannot be proxied through the OpenAI-compatible provider. "
                        "Remove server tool content and handle it client-side."
                    )

            ai_msg = AIMessage(content="\n".join(text_parts))
            if tool_calls:
                ai_msg.tool_calls = tool_calls
            msgs.append(ai_msg)

    return msgs


def convert_tool_choice(tool_choice: ToolChoice | None) -> str | None:
    """Convert an internal tool choice to the string format expected by LangChain
    ``bind_tools``."""
    if tool_choice is None:
        return None
    if tool_choice.mode == "auto":
        return "auto"
    if tool_choice.mode == "any":
        return "any"
    if tool_choice.mode == "tool":
        return tool_choice.tool_name
    return None


def bind_tools_to_model(
    chat_model: BaseChatModel,
    tools: list[ToolSpec] | None,
    tool_choice: ToolChoice | None,
) -> Runnable:
    """Bind tools to a LangChain model, returning a Runnable.

    If tools is None or empty, returns the model unchanged.
    """
    if not tools:
        return chat_model

    tool_dicts = [t.model_dump() for t in tools]
    lc_tool_choice = convert_tool_choice(tool_choice)

    kwargs: dict[str, Any] = {}
    if lc_tool_choice is not None:
        kwargs["tool_choice"] = lc_tool_choice
    return chat_model.bind_tools(tool_dicts, **kwargs)


def extract_invoke_params(request: InternalRequest) -> dict[str, Any]:
    """Extract per-request LangChain invoke parameters from the internal request."""
    params: dict[str, Any] = {"max_tokens": request.max_tokens}
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.top_k is not None:
        params["top_k"] = request.top_k
    if request.stop_sequences:
        params["stop"] = request.stop_sequences
    return params


def from_langchain_response(
    response: AIMessage,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> InternalResult:
    """Convert a LangChain ``AIMessage`` to an ``InternalResult``."""
    parts: list[Part] = []

    if isinstance(response.content, str) and response.content:
        parts.append(TextPart(text=response.content))
    elif isinstance(response.content, list):
        for p in response.content:
            if isinstance(p, str):
                parts.append(TextPart(text=p))
            elif isinstance(p, dict) and p.get("type") == "text":
                parts.append(TextPart(text=p["text"]))

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            parts.append(ToolCallPart(
                id=tc.get("id", str(uuid.uuid4())),
                name=tc["name"],
                input=tc["args"],
            ))

    usage_meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(usage_meta, dict):
        input_tokens = input_tokens or usage_meta.get("input_tokens", 0)
        output_tokens = output_tokens or usage_meta.get("output_tokens", 0)

    stop_reason = "end_turn"
    finish_reason = getattr(response, "response_metadata", {}).get("finish_reason")
    if finish_reason == "stop":
        stop_reason = "end_turn"
    elif finish_reason == "tool_calls" or (
        hasattr(response, "tool_calls") and response.tool_calls
    ):
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"

    if not parts:
        parts.append(TextPart(text=""))

    return InternalResult(
        content=parts,
        stop_reason=stop_reason,
        usage=InternalUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
