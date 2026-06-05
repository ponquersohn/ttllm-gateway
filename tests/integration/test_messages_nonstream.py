"""Non-streaming /v1/messages against the real boto3 → fake Bedrock path."""

from __future__ import annotations

import httpx


def test_nonstreaming_message(client: httpx.Client, gateway_user_token: str, bedrock_model: dict):
    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": gateway_user_token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "hello there"}],
            "max_tokens": 128,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["type"] == "message"
    assert data["role"] == "assistant"
    # Fake Bedrock echoes the user text back.
    text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    assert "hello there" in text
    assert data["stop_reason"] == "end_turn"

    usage = data["usage"]
    assert usage["input_tokens"] >= 1
    assert usage["output_tokens"] >= 1
