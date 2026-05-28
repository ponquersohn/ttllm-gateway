"""Translate between Anthropic Messages API format and LangChain messages.

No framework dependencies -- only pydantic schemas and langchain-core.
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

from ttllm.schemas.anthropic import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingBlock,
    ServerToolDefinition,
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
)

ToolChoice = ToolChoiceAuto | ToolChoiceAny | ToolChoiceTool | ToolChoiceNone


def partition_tools(
    tools: list[ToolDefinition | ServerToolDefinition] | None,
) -> tuple[list[ToolDefinition], list[ServerToolDefinition]]:
    """Separate client-defined tools from server-side tool declarations."""
    if not tools:
        return [], []
    client_tools = [t for t in tools if isinstance(t, ToolDefinition)]
    server_tools = [t for t in tools if isinstance(t, ServerToolDefinition)]
    return client_tools, server_tools


def _convert_content_to_langchain(content: str | list[ContentBlock]) -> str | list[dict]:
    """Convert Anthropic content blocks to LangChain format."""
    if isinstance(content, str):
        return content

    parts: list[dict] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.source.media_type};base64,{block.source.data}"
                },
            })
        elif isinstance(block, DocumentBlock):
            parts.append({
                "type": "text",
                "text": f"[document: {block.title or 'untitled'}]",
            })
        elif isinstance(block, ToolUseBlock):
            parts.append({
                "type": "text",
                "text": f"[tool_use: {block.name}({block.input})]",
            })
        elif isinstance(block, ToolResultBlock):
            result_text = block.content if isinstance(block.content, str) else " ".join(
                b.text for b in block.content
            )
            parts.append({"type": "text", "text": result_text})
        elif isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
            pass
    return parts


def to_langchain_messages(request: MessagesRequest) -> list[BaseMessage]:
    """Convert an Anthropic MessagesRequest into a list of LangChain messages."""
    msgs: list[BaseMessage] = []

    if request.system:
        if isinstance(request.system, str):
            msgs.append(SystemMessage(content=request.system))
        else:
            system_text = "\n".join(b.text for b in request.system)
            msgs.append(SystemMessage(content=system_text))

    for msg in request.messages:
        content = _convert_content_to_langchain(msg.content)

        if msg.role == "user":
            if isinstance(msg.content, list):
                tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
                if tool_results:
                    for tr in tool_results:
                        result_content = tr.content if isinstance(tr.content, str) else " ".join(
                            b.text for b in tr.content
                        )
                        msgs.append(ToolMessage(
                            content=result_content,
                            tool_call_id=tr.tool_use_id,
                        ))
                    non_tool = [b for b in msg.content if not isinstance(b, ToolResultBlock)]
                    if non_tool:
                        msgs.append(HumanMessage(
                            content=_convert_content_to_langchain(non_tool)
                        ))
                    continue
            msgs.append(HumanMessage(content=content))

        elif msg.role == "assistant":
            if isinstance(msg.content, list):
                tool_calls = []
                text_parts = []
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        tool_calls.append({
                            "name": block.name,
                            "args": block.input,
                            "id": block.id,
                        })
                    elif isinstance(block, TextBlock):
                        text_parts.append(block.text)

                ai_content = "\n".join(text_parts) if text_parts else ""
                ai_msg = AIMessage(content=ai_content)
                if tool_calls:
                    ai_msg.tool_calls = tool_calls
                msgs.append(ai_msg)
            else:
                msgs.append(AIMessage(content=content))

    return msgs


def convert_tool_choice(tool_choice: ToolChoice | None) -> str | None:
    """Convert Anthropic tool_choice to the string format expected by LangChain bind_tools."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, ToolChoiceNone):
        return "none"
    if isinstance(tool_choice, ToolChoiceAuto):
        return "auto"
    if isinstance(tool_choice, ToolChoiceAny):
        return "any"
    if isinstance(tool_choice, ToolChoiceTool):
        return tool_choice.name
    return None


def bind_tools_to_model(
    chat_model: BaseChatModel,
    tools: list[ToolDefinition] | None,
    tool_choice: ToolChoice | None,
) -> Runnable:
    """Bind client-defined tools to a LangChain model, returning a Runnable.

    Only accepts ToolDefinition instances (client tools). Server tools must be
    filtered out via partition_tools() before calling this function.
    """
    if not tools:
        return chat_model

    tool_dicts = [t.model_dump() for t in tools]
    lc_tool_choice = convert_tool_choice(tool_choice)

    kwargs: dict[str, Any] = {}
    if lc_tool_choice is not None:
        kwargs["tool_choice"] = lc_tool_choice
    return chat_model.bind_tools(tool_dicts, **kwargs)


def extract_invoke_params(request: MessagesRequest) -> dict[str, Any]:
    """Extract per-request LangChain invoke parameters from the Anthropic request."""
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
    model_name: str,
    request_id: uuid.UUID,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> MessagesResponse:
    """Convert a LangChain AIMessage to an Anthropic MessagesResponse."""
    content_blocks: list[ContentBlock] = []

    if isinstance(response.content, str) and response.content:
        content_blocks.append(TextBlock(text=response.content))
    elif isinstance(response.content, list):
        for part in response.content:
            if isinstance(part, str):
                content_blocks.append(TextBlock(text=part))
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    content_blocks.append(TextBlock(text=part["text"]))

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            content_blocks.append(ToolUseBlock(
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

    if not content_blocks:
        content_blocks.append(TextBlock(text=""))

    return MessagesResponse(
        id=f"msg_{request_id.hex[:24]}",
        content=content_blocks,
        model=model_name,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
