"""End-to-end prompt-caching path: cache_control -> Bedrock cachePoint -> cache usage.

Proves the full chain works against the real boto3 -> fake Bedrock path:
  client sends Anthropic `cache_control`
    -> TTLLM schema preserves it
    -> build_converse_request emits a Bedrock `cachePoint`
    -> fake Bedrock sees the cachePoint and reports cacheReadInputTokens
    -> parse_converse_response surfaces it as usage.cache_read_input_tokens

The fake Bedrock (tests/integration/fake_bedrock/app.py) attributes input tokens to
a cache read whenever a cachePoint is present, so a non-zero cache_read_input_tokens
on the response is direct evidence the marker survived translation.
"""

from __future__ import annotations

import httpx


def _invoke(client: httpx.Client, token: str, model: str, body: dict) -> httpx.Response:
    return client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": token},
        json=body,
    )


def test_cache_control_on_system_emits_cache_read(
    client: httpx.Client, gateway_user_token: str, bedrock_model: dict
):
    """A system block with cache_control should produce cache-read usage."""
    resp = _invoke(client, gateway_user_token, bedrock_model["name"], {
        "model": bedrock_model["name"],
        "system": [
            {"type": "text", "text": "Stable cacheable preamble.", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 64,
    })
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["cache_read_input_tokens"] is not None
    assert usage["cache_read_input_tokens"] > 0


def test_cache_control_on_message_emits_cache_read(
    client: httpx.Client, gateway_user_token: str, bedrock_model: dict
):
    """A message text block with cache_control should produce cache-read usage."""
    resp = _invoke(client, gateway_user_token, bedrock_model["name"], {
        "model": bedrock_model["name"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Large stable context.", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "and the variable question"},
                ],
            }
        ],
        "max_tokens": 64,
    })
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["cache_read_input_tokens"] is not None
    assert usage["cache_read_input_tokens"] > 0


def test_no_cache_control_means_no_cache_read(
    client: httpx.Client, gateway_user_token: str, bedrock_model: dict
):
    """Regression guard: without cache_control, no cachePoint, no cache read."""
    resp = _invoke(client, gateway_user_token, bedrock_model["name"], {
        "model": bedrock_model["name"],
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": 64,
    })
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    # null or 0 — never a positive cache read when nothing was marked.
    assert not usage.get("cache_read_input_tokens")
