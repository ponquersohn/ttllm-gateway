"""Internal chunk stream -> Anthropic SSE wire format.

The one place that knows the Anthropic SSE event grammar (``message_start`` ->
``ping`` -> repeated [``content_block_start`` -> ``content_block_delta``* ->
``content_block_stop``] -> ``message_delta`` -> ``message_stop``). Any provider that can
produce an ``AsyncIterator[InternalChunk]`` gets Anthropic-compatible streaming for free
by passing it through here -- this replaces two previously-independent hand-rolled
implementations (Bedrock's and LangChain's).

Providers are expected to emit a ``BLOCK_START`` chunk before any delta for that index
(Bedrock's native stream omits this for some block types; ``core/providers/bedrock/converse.py``
synthesizes it there, since that's Bedrock-shape-specific bookkeeping, not shared SSE-grammar
logic).
This module still defensively closes any block left open when the stream ends, as a
safety net.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

from ttllm.core.adapters.anthropic import _part_to_block
from ttllm.core.model import ChunkKind, InternalChunk
from ttllm.schemas.anthropic import MessagesResponse, Usage


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def encode_anthropic_sse(
    chunks: AsyncIterator[InternalChunk],
    model_name: str,
    request_id: uuid.UUID,
) -> AsyncIterator[str]:
    start_msg = MessagesResponse(
        id=f"msg_{request_id.hex[:24]}",
        content=[],
        model=model_name,
        stop_reason=None,
        usage=Usage(input_tokens=0, output_tokens=0),
    )
    yield _sse_event("message_start", {"type": "message_start", "message": start_msg.model_dump()})
    yield _sse_event("ping", {"type": "ping"})

    open_blocks: set[int] = set()
    stop_reason = "end_turn"
    usage_dict: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
    }

    async for chunk in chunks:
        if chunk.kind == ChunkKind.BLOCK_START:
            open_blocks.add(chunk.index)
            content_block = _part_to_block(chunk.part).model_dump() if chunk.part else {"type": "text", "text": ""}
            yield _sse_event("content_block_start", {
                "type": "content_block_start",
                "index": chunk.index,
                "content_block": content_block,
            })

        elif chunk.kind == ChunkKind.TEXT_DELTA:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": chunk.index,
                "delta": {"type": "text_delta", "text": chunk.text or ""},
            })

        elif chunk.kind == ChunkKind.TOOL_ARGS_DELTA:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": chunk.index,
                "delta": {"type": "input_json_delta", "partial_json": chunk.partial_json or ""},
            })

        elif chunk.kind == ChunkKind.THINKING_DELTA:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": chunk.index,
                "delta": {"type": "thinking_delta", "thinking": chunk.text or ""},
            })

        elif chunk.kind == ChunkKind.SIGNATURE_DELTA:
            yield _sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": chunk.index,
                "delta": {"type": "signature_delta", "signature": chunk.signature or ""},
            })

        elif chunk.kind == ChunkKind.BLOCK_STOP:
            open_blocks.discard(chunk.index)
            yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": chunk.index})

        elif chunk.kind == ChunkKind.MESSAGE_STOP:
            stop_reason = chunk.stop_reason or "end_turn"
            if chunk.usage:
                usage_dict = {
                    "input_tokens": chunk.usage.input_tokens,
                    "output_tokens": chunk.usage.output_tokens,
                    "cache_read_input_tokens": chunk.usage.cache_read_tokens or None,
                    "cache_creation_input_tokens": chunk.usage.cache_write_tokens or None,
                }

        elif chunk.kind == ChunkKind.ERROR:
            yield _sse_event("error", {
                "type": "error",
                "error": {"type": "api_error", "message": chunk.error_message or "Unknown error"},
            })
            return

    for idx in sorted(open_blocks):
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})

    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage_dict,
    })
    yield _sse_event("message_stop", {"type": "message_stop"})
