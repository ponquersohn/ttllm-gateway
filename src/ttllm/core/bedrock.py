"""Direct Bedrock Converse API integration — no LangChain dependency."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import boto3
from botocore.config import Config as BotocoreConfig

from ttllm.config import settings
from ttllm.schemas.anthropic import (
    ContentBlock,
    DocumentBlock,
    ImageBlock,
    Message,
    MessagesRequest,
    MessagesResponse,
    RedactedThinkingBlock,
    ServerToolDefinition,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


_CLIENT_CACHE: OrderedDict[str, Any] = OrderedDict()
_CLIENT_CACHE_MAX = 32

# Bedrock prompt prefill on large contexts can exceed botocore's 60s default read
# timeout before the first stream event arrives, so the default here is deliberately
# generous. Both timeouts are per-model tunable via config_json.
_DEFAULT_READ_TIMEOUT = 300
_DEFAULT_CONNECT_TIMEOUT = 10
_DEFAULT_MAX_ATTEMPTS = 3

# Dedicated thread pool for blocking boto3 Bedrock calls, so they don't
# contend with the asyncio default executor (which other libraries share).
_BEDROCK_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="bedrock")


def _cache_point() -> dict[str, Any]:
    """Bedrock Converse cache breakpoint marker.

    Inserted as a sibling element immediately after the block that carried an
    Anthropic `cache_control` marker, telling Bedrock to cache the prefix up to
    and including that block.
    """
    return {"cachePoint": {"type": "default"}}


def get_boto3_client(llm_model: Any) -> Any:
    config = llm_model.config_json or {}
    read_timeout = config.get("read_timeout", _DEFAULT_READ_TIMEOUT)
    connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
    max_attempts = config.get("retry_max_attempts", _DEFAULT_MAX_ATTEMPTS)
    cache_key = (
        f"{config.get('aws_profile', '')}:{config.get('aws_access_key_id', '')}:"
        f"{config.get('region', '')}:{config.get('endpoint_url', '')}:"
        f"{read_timeout}:{connect_timeout}:{max_attempts}"
    )

    if cache_key in _CLIENT_CACHE:
        _CLIENT_CACHE.move_to_end(cache_key)
        return _CLIENT_CACHE[cache_key]

    session_kwargs: dict[str, Any] = {}
    if config.get("aws_profile"):
        session_kwargs["profile_name"] = config["aws_profile"]
    if config.get("aws_access_key_id"):
        session_kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        session_kwargs["aws_secret_access_key"] = config.get("aws_secret_access_key", "")
        if config.get("aws_session_token"):
            session_kwargs["aws_session_token"] = config["aws_session_token"]
    session_kwargs["region_name"] = config.get("region", settings.provider.default_region)

    # endpoint_url lets the client target a non-default endpoint (VPC interface
    # endpoint, LocalStack, or a test double). Omitted → boto3 uses the AWS default.
    client_kwargs: dict[str, Any] = {
        "config": BotocoreConfig(
            read_timeout=read_timeout,
            connect_timeout=connect_timeout,
            retries={"mode": "standard", "max_attempts": max_attempts},
        ),
    }
    if config.get("endpoint_url"):
        client_kwargs["endpoint_url"] = config["endpoint_url"]

    client = boto3.Session(**session_kwargs).client("bedrock-runtime", **client_kwargs)

    if len(_CLIENT_CACHE) >= _CLIENT_CACHE_MAX:
        _CLIENT_CACHE.popitem(last=False)
    _CLIENT_CACHE[cache_key] = client

    return client


def _image_block_to_bedrock(block: ImageBlock) -> dict[str, Any]:
    fmt = block.source.media_type.split("/")[1] if "/" in block.source.media_type else block.source.media_type
    return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(block.source.data)}}}


def _tool_result_content_to_bedrock(content: Any) -> list[dict[str, Any]]:
    """Convert ToolResultBlock content into a Bedrock toolResult content list."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: list[dict[str, Any]] = []
    for b in content:
        if isinstance(b, TextBlock):
            parts.append({"text": b.text})
        elif isinstance(b, ImageBlock):
            parts.append(_image_block_to_bedrock(b))
        elif hasattr(b, "text"):
            parts.append({"text": b.text})
    return parts


def _convert_content_block_to_bedrock(block: ContentBlock) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"text": block.text}
    if isinstance(block, ImageBlock):
        return _image_block_to_bedrock(block)
    if isinstance(block, DocumentBlock):
        fmt = "pdf"
        if block.source.media_type and "/" in block.source.media_type:
            fmt = block.source.media_type.split("/")[1]
        return {
            "document": {
                "format": fmt,
                "name": block.title or "document",
                "source": {"bytes": base64.b64decode(block.source.data)},
            }
        }
    if isinstance(block, ToolUseBlock):
        return {"toolUse": {"toolUseId": block.id, "name": block.name, "input": block.input}}
    if isinstance(block, ToolResultBlock):
        return {
            "toolResult": {
                "toolUseId": block.tool_use_id,
                "content": _tool_result_content_to_bedrock(block.content),
                "status": "error" if block.is_error else "success",
            }
        }
    if isinstance(block, ThinkingBlock):
        return {"reasoningContent": {"reasoningText": {"text": block.thinking, "signature": block.signature}}}
    # RedactedThinkingBlock has no Bedrock equivalent; server-side tool-use blocks
    # cannot be replayed to Bedrock. Drop both rather than corrupting the request.
    if isinstance(block, (RedactedThinkingBlock, ServerToolUseBlock)):
        return None
    return {"text": str(block)}


def _convert_message_to_bedrock(msg: Message) -> dict[str, Any]:
    if isinstance(msg.content, str):
        return {"role": msg.role, "content": [{"text": msg.content}]}

    content_parts: list[dict[str, Any]] = []
    for block in msg.content:
        converted = _convert_content_block_to_bedrock(block)
        if converted is not None:
            content_parts.append(converted)
            if getattr(block, "cache_control", None):
                content_parts.append(_cache_point())

    return {"role": msg.role, "content": content_parts}


def _message_to_system_text(msg: Message) -> str:
    """Flatten a system-role message's content into plain text.

    Bedrock Converse system blocks are text-only, so non-text content blocks
    (the model would not send them on a system turn) are dropped.
    """
    if isinstance(msg.content, str):
        return msg.content
    return "\n".join(b.text for b in msg.content if isinstance(b, TextBlock))


def _convert_tools_to_bedrock(tools: list[ToolDefinition | ServerToolDefinition]) -> list[dict[str, Any]]:
    tool_specs = []
    for tool in tools:
        if isinstance(tool, ServerToolDefinition):
            continue
        spec: dict[str, Any] = {"toolSpec": {"name": tool.name, "description": tool.description}}
        input_schema = {"type": "object", "properties": tool.input_schema.properties}
        if tool.input_schema.required:
            input_schema["required"] = tool.input_schema.required
        spec["toolSpec"]["inputSchema"] = {"json": input_schema}
        tool_specs.append(spec)
        if getattr(tool, "cache_control", None):
            tool_specs.append(_cache_point())
    return tool_specs


def build_converse_request(request: MessagesRequest, llm_model: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"modelId": llm_model.provider_model_id}

    # Bedrock Converse only accepts user/assistant turns in `messages`; system
    # turns (Anthropic mid-conversation system messages) are lifted out and
    # appended to the top-level system array, preserving their order.
    messages = [
        _convert_message_to_bedrock(msg)
        for msg in request.messages
        if msg.role != "system"
    ]
    params["messages"] = messages

    system_blocks: list[dict[str, Any]] = []
    if request.system:
        if isinstance(request.system, str):
            system_blocks.append({"text": request.system})
        else:
            for block in request.system:
                system_blocks.append({"text": block.text})
                if getattr(block, "cache_control", None):
                    system_blocks.append(_cache_point())
    for msg in request.messages:
        if msg.role == "system":
            text = _message_to_system_text(msg)
            if text:
                system_blocks.append({"text": text})
    if system_blocks:
        params["system"] = system_blocks

    inference_config: dict[str, Any] = {"maxTokens": request.max_tokens}
    if request.temperature is not None:
        inference_config["temperature"] = request.temperature
    if request.top_p is not None:
        inference_config["topP"] = request.top_p
    if request.stop_sequences:
        inference_config["stopSequences"] = request.stop_sequences
    params["inferenceConfig"] = inference_config

    if request.tools:
        client_tools = [t for t in request.tools if not isinstance(t, ServerToolDefinition)]
        if client_tools:
            from ttllm.schemas.anthropic import ToolChoiceAny, ToolChoiceAuto, ToolChoiceNone, ToolChoiceTool

            # Bedrock Converse has no "none" toolChoice. To forbid tool calls,
            # omit toolConfig entirely so the model has no tools to call.
            if isinstance(request.tool_choice, ToolChoiceNone):
                pass
            else:
                tool_config: dict[str, Any] = {"tools": _convert_tools_to_bedrock(client_tools)}
                if isinstance(request.tool_choice, ToolChoiceAuto):
                    tool_config["toolChoice"] = {"auto": {}}
                elif isinstance(request.tool_choice, ToolChoiceAny):
                    tool_config["toolChoice"] = {"any": {}}
                elif isinstance(request.tool_choice, ToolChoiceTool):
                    tool_config["toolChoice"] = {"tool": {"name": request.tool_choice.name}}
                params["toolConfig"] = tool_config

    additional_fields: dict[str, Any] = {}
    if request.top_k is not None:
        additional_fields["top_k"] = request.top_k
    if request.thinking:
        additional_fields["thinking"] = request.thinking
    if additional_fields:
        params["additionalModelRequestFields"] = additional_fields

    return params


_BEDROCK_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filtered": "end_turn",
    "guardrail_intervened": "end_turn",
}


def _map_stop_reason(bedrock_reason: str) -> str:
    return _BEDROCK_STOP_REASON_MAP.get(bedrock_reason, "end_turn")


def _parse_bedrock_content_block(block: dict[str, Any]) -> ContentBlock:
    if "text" in block:
        return TextBlock(text=block["text"])
    if "toolUse" in block:
        tu = block["toolUse"]
        return ToolUseBlock(id=tu["toolUseId"], name=tu["name"], input=tu.get("input", {}))
    if "reasoningContent" in block:
        rc = block["reasoningContent"]
        if "reasoningText" in rc:
            return ThinkingBlock(thinking=rc["reasoningText"]["text"], signature=rc["reasoningText"].get("signature", ""))
        return ThinkingBlock(thinking=str(rc), signature="")
    if "image" in block:
        img = block["image"]
        fmt = img.get("format", "png")
        data = base64.b64encode(img["source"]["bytes"]).decode() if isinstance(img["source"]["bytes"], bytes) else img["source"]["bytes"]
        from ttllm.schemas.anthropic import ImageSource
        return ImageBlock(source=ImageSource(media_type=f"image/{fmt}", data=data))
    return TextBlock(text=json.dumps(block))


def _assembled_to_content_blocks(assembled: dict[int, dict[str, Any]]) -> list[ContentBlock]:
    """Rebuild Anthropic content blocks from the per-index builders accumulated while
    streaming. Tool-use input arrives as a JSON string and is parsed back to a dict."""
    blocks: list[ContentBlock] = []
    for idx in sorted(assembled):
        b = assembled[idx]
        kind = b.get("type")
        if kind == "text":
            blocks.append(TextBlock(text=b.get("text", "")))
        elif kind == "tool_use":
            raw = b.get("input_json", "") or ""
            try:
                parsed = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                parsed = {}
            blocks.append(ToolUseBlock(id=b.get("id", ""), name=b.get("name", ""), input=parsed))
        elif kind == "thinking":
            blocks.append(ThinkingBlock(thinking=b.get("thinking", ""), signature=b.get("signature", "")))
    return blocks


def parse_converse_response(
    response: dict[str, Any], model_name: str, request_id: uuid.UUID
) -> tuple[MessagesResponse, int, int]:
    """Convert a Bedrock Converse response into an Anthropic response.

    Returns ``(response, cache_read_tokens, cache_write_tokens)``. ``input_tokens``
    in the returned Usage counts only fresh input — cache-read tokens are reported
    separately via ``cache_read_input_tokens`` and billed at their own rate.
    """
    output = response.get("output", {})
    message = output.get("message", {})
    raw_content = message.get("content", [])

    content_blocks: list[ContentBlock] = []
    for block in raw_content:
        content_blocks.append(_parse_bedrock_content_block(block))

    if not content_blocks:
        content_blocks.append(TextBlock(text=""))

    stop_reason = _map_stop_reason(response.get("stopReason", "end_turn"))

    usage_data = response.get("usage", {})
    input_tokens = usage_data.get("inputTokens", 0)
    output_tokens = usage_data.get("outputTokens", 0)
    cache_read = usage_data.get("cacheReadInputTokens", 0)
    cache_write = usage_data.get("cacheWriteInputTokens", 0)

    usage = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write if cache_write else None,
        cache_read_input_tokens=cache_read if cache_read else None,
    )

    return (
        MessagesResponse(
            id=f"msg_{request_id.hex[:24]}",
            content=content_blocks,
            model=model_name,
            stop_reason=stop_reason,
            usage=usage,
        ),
        cache_read,
        cache_write,
    )


async def _converse_raw(
    request: MessagesRequest, llm_model: Any
) -> dict[str, Any]:
    """Run the blocking Bedrock Converse call on the executor; return the raw response."""
    client = get_boto3_client(llm_model)
    params = build_converse_request(request, llm_model)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_BEDROCK_EXECUTOR, lambda: client.converse(**params))


async def invoke_converse(
    request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
) -> tuple[MessagesResponse, int, int]:
    response = await _converse_raw(request, llm_model)
    return parse_converse_response(response, llm_model.name, request_id)


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


_STREAM_SENTINEL = object()


async def _aiter_event_stream(
    stream: Any, loop: asyncio.AbstractEventLoop
) -> AsyncIterator[dict[str, Any]]:
    """Asynchronously iterate a synchronous botocore EventStream.

    Each ``next()`` is run on the Bedrock executor so the event loop is never
    blocked, and events are yielded incrementally as botocore decodes them.
    StopIteration is converted to a sentinel inside the worker thread — a raw
    StopIteration must never cross the executor boundary.
    """
    iterator = iter(stream)

    def _next_event() -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return _STREAM_SENTINEL

    while True:
        event = await loop.run_in_executor(_BEDROCK_EXECUTOR, _next_event)
        if event is _STREAM_SENTINEL:
            return
        yield event


async def stream_converse(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
    usage_out: dict[str, int] | None = None,
    state: Any = None,
) -> AsyncIterator[str]:
    """Stream a Bedrock Converse response as Anthropic SSE events.

    ``usage_out`` (a plain dict) and ``state`` (a duck-typed accumulator, e.g.
    ``BedrockState``) are both optional sinks for the final token counts. When ``state`` is
    provided, the assembled content blocks and stop reason are accumulated onto it as well,
    so the full response can be rebuilt after the stream ends.
    """
    client = get_boto3_client(llm_model)
    params = build_converse_request(request, llm_model)

    loop = asyncio.get_running_loop()

    try:
        response = await loop.run_in_executor(_BEDROCK_EXECUTOR, lambda: client.converse_stream(**params))
    except Exception as exc:
        yield _sse_event("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        return

    stream = response.get("stream")
    if not stream:
        yield _sse_event("error", {"type": "error", "error": {"type": "api_error", "message": "No stream in response"}})
        return

    block_index = 0
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    raw_usage: dict[str, Any] = {}
    stop_reason = "end_turn"
    # Per-index builders for rebuilding the full response from the streamed deltas.
    # Each entry: {"type": "text"|"tool_use"|"thinking", ...accumulators...}.
    assembled: dict[int, dict[str, Any]] = {}
    # Block indices we've opened with a content_block_start but not yet closed
    # with a content_block_stop. Bedrock Converse only sends contentBlockStart
    # for tool-use blocks; for text/reasoning it jumps straight to
    # contentBlockDelta. The Anthropic wire protocol requires a
    # content_block_start before any delta for that index, so we open the block
    # lazily on the first delta when Bedrock omits it — and make sure to close
    # any still-open block before message_delta.
    open_blocks: set[int] = set()

    try:
        async for event in _aiter_event_stream(stream, loop):
            if "messageStart" in event:
                start_msg = MessagesResponse(
                    id=f"msg_{request_id.hex[:24]}",
                    content=[],
                    model=llm_model.name,
                    stop_reason=None,
                    usage=Usage(input_tokens=0, output_tokens=0),
                )
                yield _sse_event("message_start", {"type": "message_start", "message": start_msg.model_dump()})
                yield _sse_event("ping", {"type": "ping"})

            elif "contentBlockStart" in event:
                cbs = event["contentBlockStart"]
                idx = cbs.get("contentBlockIndex", block_index)
                block_index = idx
                open_blocks.add(idx)
                start_block = cbs.get("start", {})

                if "toolUse" in start_block:
                    tu = start_block["toolUse"]
                    assembled[idx] = {"type": "tool_use", "id": tu.get("toolUseId", ""), "name": tu.get("name", ""), "input_json": ""}
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "tool_use", "id": tu.get("toolUseId", ""), "name": tu.get("name", ""), "input": {}},
                    })
                elif "reasoningContent" in start_block:
                    assembled[idx] = {"type": "thinking", "thinking": "", "signature": ""}
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    })
                else:
                    assembled[idx] = {"type": "text", "text": ""}
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    })

            elif "contentBlockDelta" in event:
                cbd = event["contentBlockDelta"]
                idx = cbd.get("contentBlockIndex", block_index)
                delta = cbd.get("delta", {})

                # Open the block if Bedrock didn't send a contentBlockStart for it
                # (it omits the start event for text and reasoning blocks).
                if idx not in open_blocks:
                    open_blocks.add(idx)
                    if "reasoningContent" in delta:
                        opening_block = {"type": "thinking", "thinking": "", "signature": ""}
                    else:
                        opening_block = {"type": "text", "text": ""}
                    assembled[idx] = dict(opening_block)
                    yield _sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": opening_block,
                    })

                if "text" in delta:
                    assembled.setdefault(idx, {"type": "text", "text": ""})["text"] += delta["text"]
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": delta["text"]},
                    })
                elif "toolUse" in delta:
                    assembled.setdefault(idx, {"type": "tool_use", "id": "", "name": "", "input_json": ""})["input_json"] += delta["toolUse"].get("input", "")
                    yield _sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": delta["toolUse"].get("input", "")},
                    })
                elif "reasoningContent" in delta:
                    rc = delta["reasoningContent"]
                    if "text" in rc:
                        assembled.setdefault(idx, {"type": "thinking", "thinking": "", "signature": ""})["thinking"] += rc["text"]
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "thinking_delta", "thinking": rc["text"]},
                        })
                    elif "signature" in rc:
                        assembled.setdefault(idx, {"type": "thinking", "thinking": "", "signature": ""})["signature"] = rc["signature"]
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "signature_delta", "signature": rc["signature"]},
                        })

            elif "contentBlockStop" in event:
                idx = event["contentBlockStop"].get("contentBlockIndex", block_index)
                open_blocks.discard(idx)
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})
                block_index = idx + 1

            elif "messageStop" in event:
                stop_reason = _map_stop_reason(event["messageStop"].get("stopReason", "end_turn"))

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                raw_usage = usage
                input_tokens = usage.get("inputTokens", 0)
                output_tokens = usage.get("outputTokens", 0)
                cache_read = usage.get("cacheReadInputTokens", 0)
                cache_write = usage.get("cacheWriteInputTokens", 0)
    except Exception as exc:
        yield _sse_event("error", {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
        return
    finally:
        if usage_out is not None:
            usage_out["input_tokens"] = input_tokens
            usage_out["output_tokens"] = output_tokens
            usage_out["cache_read_tokens"] = cache_read
            usage_out["cache_write_tokens"] = cache_write
        if state is not None:
            state.input_tokens = input_tokens
            state.output_tokens = output_tokens
            state.cache_read_tokens = cache_read
            state.cache_write_tokens = cache_write
            state.raw_usage = raw_usage
            state.stop_reason = stop_reason
            state.content_blocks = _assembled_to_content_blocks(assembled)

    # Close any block we opened lazily that Bedrock never sent a stop for, so
    # the client never sees message_delta with a content block still open.
    for idx in sorted(open_blocks):
        yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})
    open_blocks.clear()

    yield _sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"type": "message_delta", "stop_reason": stop_reason, "stop_sequence": None},
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read or None,
            "cache_creation_input_tokens": cache_write or None,
        },
    })
    yield _sse_event("message_stop", {"type": "message_stop"})
