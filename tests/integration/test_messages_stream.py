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


def test_streaming_persists_response_body_and_cost(
    client: httpx.Client, admin_headers: dict, gateway_user_token: str, bedrock_model: dict
):
    """Streamed requests now persist the assembled response body to audit_log_bodies and
    record an authoritative cost + provider_metadata, matching non-streaming."""
    marker = "persist this stream"
    with client.stream(
        "POST",
        "/anthropic/v1/messages",
        headers={"x-api-key": gateway_user_token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": marker}],
            "max_tokens": 128,
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200, resp.read().decode()
        resp.read()

    # Find the audit row for this streamed request.
    logs = client.get(
        "/admin/audit-logs",
        headers=admin_headers,
        params={"model_id": bedrock_model["id"], "limit": 50},
    )
    assert logs.status_code == 200, logs.text
    items = logs.json()["items"]
    assert items
    latest = items[0]
    assert latest["total_cost"] is not None
    assert latest["provider_metadata"]["provider"] == "bedrock"

    # The assembled streamed response was saved as response_body.
    body = client.get(f"/admin/audit-logs/{latest['id']}/body", headers=admin_headers)
    assert body.status_code == 200, body.text
    response_body = body.json()["response_body"]
    assert response_body is not None
    text = "".join(
        b.get("text", "") for b in response_body.get("content", []) if b.get("type") == "text"
    )
    assert marker in text
