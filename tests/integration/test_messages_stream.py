"""Streaming /v1/messages: exercises boto3 converse_stream → AWS event-stream → SSE."""

from __future__ import annotations

import json

import httpx


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.strip().split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln]
        if len(lines) < 2:
            continue
        etype = lines[0].replace("event: ", "")
        data = json.loads(lines[1].replace("data: ", ""))
        events.append((etype, data))
    return events


def test_streaming_message(client: httpx.Client, gateway_user_token: str, bedrock_model: dict):
    with client.stream(
        "POST",
        "/anthropic/v1/messages",
        headers={"x-api-key": gateway_user_token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 128,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200, resp.read().decode()
        raw = "".join(resp.iter_text())

    events = _parse_sse(raw)
    types = [t for t, _ in events]

    assert types[0] == "message_start"
    assert "content_block_delta" in types
    assert "message_delta" in types
    assert types[-1] == "message_stop"

    # Reconstruct streamed text from text deltas.
    text = "".join(
        d["delta"].get("text", "")
        for t, d in events
        if t == "content_block_delta" and d["delta"].get("type") == "text_delta"
    )
    assert "stream please" in text

    # Final message_delta carries full usage (the streaming-parity fix).
    delta = next(d for t, d in events if t == "message_delta")
    assert delta["usage"]["output_tokens"] >= 1
    assert delta["usage"]["input_tokens"] >= 1
