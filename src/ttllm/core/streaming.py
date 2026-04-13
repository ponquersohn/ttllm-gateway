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
    accumulated_text = ""
    input_tokens = 0
    output_tokens = 0

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

    # content_block_start
    yield _sse_event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    chunk_index = 0
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
            accumulated_text += text
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            })

        # Try to extract token counts from the final chunk
        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta and isinstance(usage_meta, dict):
            input_tokens = usage_meta.get("input_tokens", input_tokens)
            output_tokens = usage_meta.get("output_tokens", output_tokens)

        chunk_index += 1

    # Store final token counts for the caller
    if token_usage is not None:
        token_usage["input_tokens"] = input_tokens
        token_usage["output_tokens"] = output_tokens

    # content_block_stop
    yield _sse_event("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })

    # message_delta with final usage
    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })

    # message_stop
    yield _sse_event("message_stop", {"type": "message_stop"})


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
