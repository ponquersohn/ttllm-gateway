"""Unit test: our event-stream encoder must round-trip through botocore's decoder.

This de-risks the fake Bedrock streaming server — if these frames decode cleanly here,
boto3's real EventStream parser will accept them too.
"""

from __future__ import annotations

import json

from botocore.eventstream import EventStreamBuffer

from tests.integration.fake_bedrock.eventstream_encoder import (
    encode_converse_stream,
    encode_event,
)


def _decode(body: bytes):
    buf = EventStreamBuffer()
    buf.add_data(body)
    out = []
    for msg in buf:
        payload = json.loads(msg.payload) if msg.payload else {}
        out.append((msg.headers.get(":event-type"), payload))
    return out


def test_single_event_round_trips():
    body = encode_event("messageStart", {"role": "assistant"})
    events = _decode(body)
    assert events == [("messageStart", {"role": "assistant"})]


def test_full_converse_stream_round_trips():
    body = encode_converse_stream(
        "Hello world from fake",
        input_tokens=10,
        output_tokens=5,
        cache_read=3,
        chunk_size=5,
    )
    events = _decode(body)
    types = [t for t, _ in events]

    assert types[0] == "messageStart"
    assert types[-2:] == ["messageStop", "metadata"]
    assert "contentBlockStop" in types

    text = "".join(p["delta"]["text"] for t, p in events if t == "contentBlockDelta")
    assert text == "Hello world from fake"

    usage = events[-1][1]["usage"]
    assert usage["inputTokens"] == 10
    assert usage["outputTokens"] == 5
    assert usage["cacheReadInputTokens"] == 3
