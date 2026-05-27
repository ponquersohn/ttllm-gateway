"""SSE stream formatting for Anthropic-compatible streaming responses."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from ttllm.schemas.anthropic import (
    MessagesResponse,
    TextBlock,
    Usage,
)


async def format_sse_stream(
    langchain_stream: AsyncIterator,
    model_name: str,
    request_id: uuid.UUID,
    token_usage: dict[str, int] | None = None,
) -> AsyncIterator[str]:
    """Convert a LangChain astream into Anthropic SSE events.

    Yields SSE-formatted strings: "event: <type>\\ndata: <json>\\n\\n"
    """
    input_tokens = 0
    output_tokens = 0
    block_index = 0
    has_text_block = False
    has_tool_use = False
    # Track tool calls we've already started (by id)
    started_tool_calls: set[str] = set()

    # message_start event
    start_message = MessagesResponse(
        id=f"msg_{request_id.hex[:24]}",
        content=[],
        model=model_name,
        stop_reason=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )
    yield _sse_event("message_start", {"type": "message_start", "message": start_message.model_dump()})

    # ping
    yield _sse_event("ping", {"type": "ping"})

    async for chunk in langchain_stream:
        # Extract text from chunk
        text = ""
        if hasattr(chunk, "content"):
            if isinstance(chunk.content, str):
                text = chunk.content
            elif isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text += part.get("text", "")
                    elif isinstance(part, str):
                        text += part

        if text:
            if not has_text_block:
                has_text_block = True
                yield _sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        # Handle tool call chunks
        tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
        if tool_call_chunks:
            for tc_chunk in tool_call_chunks:
                tc_id = tc_chunk.get("id")
                tc_name = tc_chunk.get("name")
                tc_args = tc_chunk.get("args", "")

                if tc_id and tc_id not in started_tool_calls:
                    # Close text block if open
                    if has_text_block and not started_tool_calls:
                        yield _sse_event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": block_index,
                        })
                        block_index += 1

                    started_tool_calls.add(tc_id)
                    has_tool_use = True
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "tool_use", "id": tc_id, "name": tc_name or "", "input": {}},
                    })

                if tc_args:
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "input_json_delta", "partial_json": tc_args},
                    })

        # Try to extract token counts from the final chunk
        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta and isinstance(usage_meta, dict):
            input_tokens = usage_meta.get("input_tokens", input_tokens)
            output_tokens = usage_meta.get("output_tokens", output_tokens)

    # Store final token counts for the caller
    if token_usage is not None:
        token_usage["input_tokens"] = input_tokens
        token_usage["output_tokens"] = output_tokens

    # Close the last open content block
    if has_text_block or has_tool_use:
        yield _sse_event("content_block_stop", {
            "type": "content_block_stop",
            "index": block_index,
        })

    stop_reason = "tool_use" if has_tool_use else "end_turn"

    # message_delta with final usage
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
