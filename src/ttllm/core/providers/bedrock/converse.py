"""Direct Bedrock Converse API integration — no LangChain dependency.

Translates between the internal representation (``ttllm.core.model``) and Bedrock's
native Converse/ConverseStream wire shapes. Carries zero dependency on the Anthropic wire
schema — see ``ttllm.core.adapters.anthropic`` for that half of the translation.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncIterator

import boto3
from botocore.config import Config as BotocoreConfig

from ttllm.config import settings
from ttllm.core.errors import ServerToolError
from ttllm.core.model import (
    ChunkKind,
    DocumentPart,
    ImagePart,
    InternalChunk,
    InternalMessage,
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
    ToolResultPart,
    ToolSpec,
)


_CLIENT_CACHE: OrderedDict[str, Any] = OrderedDict()
_CLIENT_CACHE_MAX = 32

logger = logging.getLogger(__name__)

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

    Inserted as a sibling element immediately after the block that carried an internal
    ``cache_control`` marker, telling Bedrock to cache the prefix up to and including
    that block.
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


def _image_part_to_bedrock(part: ImagePart) -> dict[str, Any]:
    fmt = part.media_type.split("/")[1] if "/" in part.media_type else part.media_type
    return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(part.data)}}}


def _tool_result_content_to_bedrock(content: list[TextPart | ImagePart]) -> list[dict[str, Any]]:
    """Convert a ToolResultPart's content into a Bedrock toolResult content list."""
    parts: list[dict[str, Any]] = []
    for p in content:
        if isinstance(p, TextPart):
            parts.append({"text": p.text})
        elif isinstance(p, ImagePart):
            parts.append(_image_part_to_bedrock(p))
    return parts


def _convert_part_to_bedrock(part: Part) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"text": part.text}
    if isinstance(part, ImagePart):
        return _image_part_to_bedrock(part)
    if isinstance(part, DocumentPart):
        fmt = "pdf"
        if part.media_type and "/" in part.media_type:
            fmt = part.media_type.split("/")[1]
        return {
            "document": {
                "format": fmt,
                "name": part.title or "document",
                "source": {"bytes": base64.b64decode(part.data)},
            }
        }
    if isinstance(part, ToolCallPart):
        return {"toolUse": {"toolUseId": part.id, "name": part.name, "input": part.input}}
    if isinstance(part, ToolResultPart):
        return {
            "toolResult": {
                "toolUseId": part.tool_call_id,
                "content": _tool_result_content_to_bedrock(part.content),
                "status": "error" if part.is_error else "success",
            }
        }
    if isinstance(part, ThinkingPart):
        return {"reasoningContent": {"reasoningText": {"text": part.text, "signature": part.signature}}}
    # RedactedThinkingPart has no Bedrock equivalent -- Anthropic itself defines this
    # content as opaque/non-replayable, so there's no data being lost by dropping it.
    if isinstance(part, RedactedThinkingPart):
        return None
    if isinstance(part, (ServerToolCallPart, ServerToolResultPart)):
        raise ServerToolError(
            "Server-side tools cannot be proxied through the Bedrock provider. "
            "Remove server tool content and handle it client-side."
        )
    return {"text": str(part)}


def _convert_message_to_bedrock(msg: InternalMessage) -> dict[str, Any]:
    content_parts: list[dict[str, Any]] = []
    for part in msg.content:
        converted = _convert_part_to_bedrock(part)
        if converted is not None:
            content_parts.append(converted)
            if part.cache_control:
                content_parts.append(_cache_point())

    return {"role": msg.role, "content": content_parts}


def _convert_tools_to_bedrock(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    tool_specs = []
    for tool in tools:
        spec: dict[str, Any] = {"toolSpec": {"name": tool.name, "description": tool.description}}
        input_schema = {
            "type": "object",
            "properties": tool.input_schema.get("properties", {}),
        }
        if tool.input_schema.get("required"):
            input_schema["required"] = tool.input_schema["required"]
        spec["toolSpec"]["inputSchema"] = {"json": input_schema}
        tool_specs.append(spec)
        if tool.cache_control:
            tool_specs.append(_cache_point())
    return tool_specs


def build_converse_request(request: InternalRequest, llm_model: Any) -> dict[str, Any]:
    if request.server_tools:
        raise ServerToolError(
            "Server-side tools cannot be proxied through the Bedrock provider. "
            "Remove server tool definitions and handle them client-side."
        )

    params: dict[str, Any] = {"modelId": llm_model.provider_model_id}
    params["messages"] = [_convert_message_to_bedrock(msg) for msg in request.messages]

    if request.system:
        system_blocks: list[dict[str, Any]] = []
        for part in request.system:
            system_blocks.append({"text": part.text})
            if part.cache_control:
                system_blocks.append(_cache_point())
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
        # Bedrock Converse has no "none" toolChoice. To forbid tool calls, omit
        # toolConfig entirely so the model has no tools to call.
        if not (request.tool_choice and request.tool_choice.mode == "none"):
            tool_config: dict[str, Any] = {"tools": _convert_tools_to_bedrock(request.tools)}
            if request.tool_choice:
                mode = request.tool_choice.mode
                if mode == "auto":
                    tool_config["toolChoice"] = {"auto": {}}
                elif mode == "any":
                    tool_config["toolChoice"] = {"any": {}}
                elif mode == "tool":
                    tool_config["toolChoice"] = {"tool": {"name": request.tool_choice.tool_name}}
            params["toolConfig"] = tool_config

    additional_fields: dict[str, Any] = {}
    if request.top_k is not None:
        additional_fields["top_k"] = request.top_k
    if request.thinking and request.thinking.enabled:
        if request.thinking.type == "adaptive":
            additional_fields["thinking"] = {"type": "adaptive"}
        else:
            additional_fields["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.thinking.budget_tokens,
            }
    if additional_fields:
        params["additionalModelRequestFields"] = additional_fields

    return params


_BEDROCK_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    # Both indicate a guardrail (content filter or denied topic/word) cut the response
    # short. Anthropic's own "refusal" stop_reason is the closest honest signal for
    # this -- folding it into "end_turn" would tell the caller the model finished
    # normally when it was actually blocked mid-generation.
    "content_filtered": "refusal",
    "guardrail_intervened": "refusal",
}


def _map_stop_reason(bedrock_reason: str) -> str:
    return _BEDROCK_STOP_REASON_MAP.get(bedrock_reason, "end_turn")


def _parse_bedrock_content_block(block: dict[str, Any]) -> Part:
    if "text" in block:
        return TextPart(text=block["text"])
    if "toolUse" in block:
        tu = block["toolUse"]
        return ToolCallPart(id=tu["toolUseId"], name=tu["name"], input=tu.get("input", {}))
    if "reasoningContent" in block:
        rc = block["reasoningContent"]
        if "reasoningText" in rc:
            return ThinkingPart(text=rc["reasoningText"]["text"], signature=rc["reasoningText"].get("signature", ""))
        return ThinkingPart(text=str(rc), signature="")
    if "image" in block:
        img = block["image"]
        fmt = img.get("format", "png")
        data = base64.b64encode(img["source"]["bytes"]).decode() if isinstance(img["source"]["bytes"], bytes) else img["source"]["bytes"]
        return ImagePart(media_type=f"image/{fmt}", data=data)
    return TextPart(text=json.dumps(block))


def _assembled_to_parts(assembled: dict[int, dict[str, Any]]) -> list[Part]:
    """Rebuild internal parts from the per-index builders accumulated while streaming.
    Tool-use input arrives as a JSON string and is parsed back to a dict."""
    parts: list[Part] = []
    for idx in sorted(assembled):
        b = assembled[idx]
        kind = b.get("type")
        if kind == "text":
            parts.append(TextPart(text=b.get("text", "")))
        elif kind == "tool_use":
            raw = b.get("input_json", "") or ""
            try:
                parsed = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                parsed = {}
            parts.append(ToolCallPart(id=b.get("id", ""), name=b.get("name", ""), input=parsed))
        elif kind == "thinking":
            parts.append(ThinkingPart(text=b.get("thinking", ""), signature=b.get("signature", "")))
    return parts


def parse_converse_response(response: dict[str, Any]) -> InternalResult:
    """Convert a Bedrock Converse response into the internal representation."""
    output = response.get("output", {})
    message = output.get("message", {})
    raw_content = message.get("content", [])

    parts: list[Part] = [_parse_bedrock_content_block(block) for block in raw_content]
    if not parts:
        parts.append(TextPart(text=""))

    stop_reason = _map_stop_reason(response.get("stopReason", "end_turn"))

    usage_data = response.get("usage", {})
    usage = InternalUsage(
        input_tokens=usage_data.get("inputTokens", 0),
        output_tokens=usage_data.get("outputTokens", 0),
        cache_read_tokens=usage_data.get("cacheReadInputTokens", 0),
        cache_write_tokens=usage_data.get("cacheWriteInputTokens", 0),
    )

    return InternalResult(content=parts, stop_reason=stop_reason, usage=usage)


async def _converse_raw(request: InternalRequest, llm_model: Any) -> dict[str, Any]:
    """Run the blocking Bedrock Converse call on the executor; return the raw response."""
    client = get_boto3_client(llm_model)
    params = build_converse_request(request, llm_model)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_BEDROCK_EXECUTOR, lambda: client.converse(**params))


async def invoke_converse(request: InternalRequest, llm_model: Any) -> InternalResult:
    response = await _converse_raw(request, llm_model)
    return parse_converse_response(response)


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


async def stream_converse_chunks(
    request: InternalRequest,
    llm_model: Any,
    request_id: uuid.UUID,
    state: Any = None,
) -> AsyncIterator[InternalChunk]:
    """Stream a Bedrock Converse response as internal chunks.

    ``state`` (a duck-typed accumulator, e.g. ``BedrockState``) is an optional sink: when
    provided, the final token counts, assembled parts and stop reason are accumulated onto
    it, so the full response can be rebuilt after the stream ends.
    """
    client = get_boto3_client(llm_model)
    params = build_converse_request(request, llm_model)

    loop = asyncio.get_running_loop()

    try:
        response = await loop.run_in_executor(_BEDROCK_EXECUTOR, lambda: client.converse_stream(**params))
    except Exception as exc:
        logger.exception("Bedrock stream failed for request %s", request_id)
        if state is not None:
            state.error = exc
        yield InternalChunk(kind=ChunkKind.ERROR, error_message=str(exc))
        return

    stream = response.get("stream")
    if not stream:
        exc = RuntimeError("No stream in response")
        logger.error("Bedrock stream failed for request %s: no stream in response", request_id)
        if state is not None:
            state.error = exc
        yield InternalChunk(kind=ChunkKind.ERROR, error_message=str(exc))
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
    # Block indices we've opened with a BLOCK_START but not yet closed with a
    # BLOCK_STOP. Bedrock Converse only sends contentBlockStart for tool-use blocks;
    # for text/reasoning it jumps straight to contentBlockDelta. The Anthropic wire
    # protocol (and our own InternalChunk contract) requires a block-start before any
    # delta for that index, so we open the block lazily on the first delta when
    # Bedrock omits it -- and make sure to close any still-open block at the end.
    open_blocks: set[int] = set()

    try:
        async for event in _aiter_event_stream(stream, loop):
            if "messageStart" in event:
                continue

            elif "contentBlockStart" in event:
                cbs = event["contentBlockStart"]
                idx = cbs.get("contentBlockIndex", block_index)
                block_index = idx
                open_blocks.add(idx)
                start_block = cbs.get("start", {})

                if "toolUse" in start_block:
                    tu = start_block["toolUse"]
                    assembled[idx] = {"type": "tool_use", "id": tu.get("toolUseId", ""), "name": tu.get("name", ""), "input_json": ""}
                    yield InternalChunk(
                        kind=ChunkKind.BLOCK_START, index=idx,
                        part=ToolCallPart(id=tu.get("toolUseId", ""), name=tu.get("name", ""), input={}),
                    )
                elif "reasoningContent" in start_block:
                    assembled[idx] = {"type": "thinking", "thinking": "", "signature": ""}
                    yield InternalChunk(kind=ChunkKind.BLOCK_START, index=idx, part=ThinkingPart(text="", signature=""))
                else:
                    assembled[idx] = {"type": "text", "text": ""}
                    yield InternalChunk(kind=ChunkKind.BLOCK_START, index=idx, part=TextPart(text=""))

            elif "contentBlockDelta" in event:
                cbd = event["contentBlockDelta"]
                idx = cbd.get("contentBlockIndex", block_index)
                delta = cbd.get("delta", {})

                # Open the block if Bedrock didn't send a contentBlockStart for it
                # (it omits the start event for text and reasoning blocks).
                if idx not in open_blocks:
                    open_blocks.add(idx)
                    if "reasoningContent" in delta:
                        assembled[idx] = {"type": "thinking", "thinking": "", "signature": ""}
                        yield InternalChunk(kind=ChunkKind.BLOCK_START, index=idx, part=ThinkingPart(text="", signature=""))
                    else:
                        assembled[idx] = {"type": "text", "text": ""}
                        yield InternalChunk(kind=ChunkKind.BLOCK_START, index=idx, part=TextPart(text=""))

                if "text" in delta:
                    assembled.setdefault(idx, {"type": "text", "text": ""})["text"] += delta["text"]
                    yield InternalChunk(kind=ChunkKind.TEXT_DELTA, index=idx, text=delta["text"])
                elif "toolUse" in delta:
                    partial = delta["toolUse"].get("input", "")
                    assembled.setdefault(idx, {"type": "tool_use", "id": "", "name": "", "input_json": ""})["input_json"] += partial
                    yield InternalChunk(kind=ChunkKind.TOOL_ARGS_DELTA, index=idx, partial_json=partial)
                elif "reasoningContent" in delta:
                    rc = delta["reasoningContent"]
                    if "text" in rc:
                        assembled.setdefault(idx, {"type": "thinking", "thinking": "", "signature": ""})["thinking"] += rc["text"]
                        yield InternalChunk(kind=ChunkKind.THINKING_DELTA, index=idx, text=rc["text"])
                    elif "signature" in rc:
                        assembled.setdefault(idx, {"type": "thinking", "thinking": "", "signature": ""})["signature"] = rc["signature"]
                        yield InternalChunk(kind=ChunkKind.SIGNATURE_DELTA, index=idx, signature=rc["signature"])

            elif "contentBlockStop" in event:
                idx = event["contentBlockStop"].get("contentBlockIndex", block_index)
                open_blocks.discard(idx)
                yield InternalChunk(kind=ChunkKind.BLOCK_STOP, index=idx)
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
        logger.exception("Bedrock stream failed for request %s", request_id)
        if state is not None:
            state.error = exc
        yield InternalChunk(kind=ChunkKind.ERROR, error_message=str(exc))
        return
    finally:
        if state is not None:
            state.input_tokens = input_tokens
            state.output_tokens = output_tokens
            state.cache_read_tokens = cache_read
            state.cache_write_tokens = cache_write
            state.raw_usage = raw_usage
            state.stop_reason = stop_reason
            state.content_parts = _assembled_to_parts(assembled)

    # Close any block we opened lazily that Bedrock never sent a stop for, so the
    # consumer never sees message_stop with a content block still open.
    for idx in sorted(open_blocks):
        yield InternalChunk(kind=ChunkKind.BLOCK_STOP, index=idx)
    open_blocks.clear()

    yield InternalChunk(
        kind=ChunkKind.MESSAGE_STOP,
        stop_reason=stop_reason,
        usage=InternalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        ),
    )
