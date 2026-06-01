"""End-to-end bootstrap: admin login → create user → model → assign → mint token."""

from __future__ import annotations

import httpx


def test_admin_can_login(admin_token: str):
    assert isinstance(admin_token, str) and admin_token


def test_weak_password_rejected_with_400(client: httpx.Client, admin_headers: dict):
    """A password failing the policy returns 400 (not 500)."""
    resp = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Weak", "email": "weak-pw@example.com", "password": "weak"},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["error"]["type"] == "invalid_request"


def test_model_created_with_endpoint_url(bedrock_model: dict):
    assert bedrock_model["provider"] == "bedrock"
    # config_json secrets are redacted in responses, but the model exists with an id.
    assert bedrock_model["id"]


def test_gateway_token_minted(gateway_user_token: str):
    assert isinstance(gateway_user_token, str) and gateway_user_token


def test_full_chain_to_message(client: httpx.Client, gateway_user_token: str, bedrock_model: dict):
    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": gateway_user_token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
