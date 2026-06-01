"""Encoder for the AWS event-stream framing (``application/vnd.amazon.eventstream``).

botocore ships only a *decoder* (``botocore.eventstream``); a fake Bedrock server needs
to *produce* frames, so we implement the inverse here.

Frame layout (all integers big-endian):

    [ total_length    : u32 ]
    [ headers_length  : u32 ]
    [ prelude_crc     : u32 ]   # crc32 of the first 8 bytes
    [ headers         : headers_length bytes ]
    [ payload         : total_length - headers_length - 16 bytes ]
    [ message_crc     : u32 ]   # crc32 of everything before it (prelude + headers + payload)

Each header:

    [ name_len : u8 ][ name : name_len bytes ][ type : u8 ][ value ... ]

We only emit string headers (type 7), whose value is ``[ value_len : u16 ][ bytes ]``.
"""

from __future__ import annotations

import json
import struct
from binascii import crc32

_HEADER_TYPE_STRING = 7


def _encode_string_header(name: str, value: str) -> bytes:
    name_b = name.encode("utf-8")
    value_b = value.encode("utf-8")
    return (
        struct.pack("!B", len(name_b))
        + name_b
        + struct.pack("!B", _HEADER_TYPE_STRING)
        + struct.pack("!H", len(value_b))
        + value_b
    )


def encode_message(headers: dict[str, str], payload: bytes) -> bytes:
    """Encode a single event-stream message from string headers + a payload."""
    header_bytes = b"".join(_encode_string_header(k, v) for k, v in headers.items())

    headers_length = len(header_bytes)
    total_length = 16 + headers_length + len(payload)  # 12 prelude + 4 message crc

    prelude = struct.pack("!II", total_length, headers_length)
    prelude_crc = crc32(prelude) & 0xFFFFFFFF
    prelude_with_crc = prelude + struct.pack("!I", prelude_crc)

    message_wo_crc = prelude_with_crc + header_bytes + payload
    message_crc = crc32(message_wo_crc) & 0xFFFFFFFF

    return message_wo_crc + struct.pack("!I", message_crc)


def encode_event(event_type: str, payload: dict) -> bytes:
    """Encode a normal Bedrock stream event (``:event-type`` = the Converse event name)."""
    headers = {
        ":event-type": event_type,
        ":content-type": "application/json",
        ":message-type": "event",
    }
    return encode_message(headers, json.dumps(payload).encode("utf-8"))


def encode_converse_stream(text: str, *, input_tokens: int, output_tokens: int,
                           cache_read: int = 0, cache_write: int = 0,
                           chunk_size: int = 4) -> bytes:
    """Build a full Converse streaming response body for a single text block.

    Emits: messageStart, contentBlockDelta(text)*, contentBlockStop, messageStop, metadata.
    """
    frames = [encode_event("messageStart", {"role": "assistant"})]

    for i in range(0, len(text), chunk_size) or [0]:
        chunk = text[i : i + chunk_size]
        frames.append(encode_event("contentBlockDelta", {
            "contentBlockIndex": 0,
            "delta": {"text": chunk},
        }))

    frames.append(encode_event("contentBlockStop", {"contentBlockIndex": 0}))
    frames.append(encode_event("messageStop", {"stopReason": "end_turn"}))

    usage: dict = {"inputTokens": input_tokens, "outputTokens": output_tokens}
    if cache_read:
        usage["cacheReadInputTokens"] = cache_read
    if cache_write:
        usage["cacheWriteInputTokens"] = cache_write
    frames.append(encode_event("metadata", {"usage": usage}))

    return b"".join(frames)
