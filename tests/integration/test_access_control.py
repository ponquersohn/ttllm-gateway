"""Access-control e2e: bad key → 401, unassigned model → 403."""

from __future__ import annotations

import os
import time

import httpx


def _unique(prefix: str) -> str:
    return f"{prefix}-{os.getpid()}-{time.monotonic_ns()}"


def test_missing_or_bad_key_unauthorized(client: httpx.Client, bedrock_model: dict):
    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": "not-a-valid-token"},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 401, resp.text


def test_unassigned_model_forbidden(client: httpx.Client, admin_headers: dict, bedrock_model: dict):
    # New user with a gateway token but NO assignment to the model.
    email = f"{_unique('it-noaccess')}@example.com"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "No Access", "email": email, "password": "NoAccess1!"},
    )
    assert u.status_code == 201, u.text
    user_id = u.json()["id"]

    # Grant llm.invoke so the 403 below is specifically about the missing model
    # assignment, not a missing permission.
    p = client.post(
        f"/admin/users/{user_id}/permissions",
        headers=admin_headers,
        json={"permissions": ["llm.invoke"]},
    )
    assert p.status_code == 201, p.text

    t = client.post(
        "/admin/tokens",
        headers=admin_headers,
        json={"user_id": user_id, "permissions": ["llm.invoke"]},
    )
    assert t.status_code == 201, t.text
    token = t.json()["access_token"]

    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 403, resp.text


def test_non_positive_max_tokens_rejected(client: httpx.Client, gateway_user_token: str, bedrock_model: dict):
    """A negative max_tokens would reserve a negative quota amount and let exhausted
    buckets admit requests, so it must never reach the quota system."""
    for max_tokens in (0, -1_000_000):
        resp = client.post(
            "/anthropic/v1/messages",
            headers={"x-api-key": gateway_user_token},
            json={
                "model": bedrock_model["name"],
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": max_tokens,
            },
        )
        assert resp.status_code == 422, resp.text


def test_gateway_api_key_cannot_mint_tokens(client: httpx.Client, gateway_user_token: str):
    """A leaked llm.invoke key must not turn itself into a broader or longer-lived credential."""
    headers = {"Authorization": f"Bearer {gateway_user_token}"}
    resp = client.post("/me/tokens", headers=headers, json={"permissions": ["llm.invoke"], "ttl_days": 365})
    assert resp.status_code == 403, resp.text
    assert client.get("/me/tokens", headers=headers).status_code == 403


def test_management_api_token_cannot_widen_itself(client: httpx.Client, admin_headers: dict):
    t = client.post("/admin/tokens", headers=admin_headers, json={"permissions": ["token.create"]})
    assert t.status_code == 201, t.text
    narrow = {"Authorization": f"Bearer {t.json()['access_token']}"}

    wider = client.post("/admin/tokens", headers=narrow, json={"permissions": ["token.create", "secret.view"]})
    assert wider.status_code == 403, wider.text
    gateway = client.post("/admin/tokens", headers=narrow, json={})
    assert gateway.status_code == 403, gateway.text

    same = client.post("/admin/tokens", headers=narrow, json={"permissions": ["token.create"]})
    assert same.status_code == 201, same.text


def test_login_session_can_still_mint_tokens(client: httpx.Client, admin_headers: dict):
    resp = client.post("/me/tokens", headers=admin_headers, json={"label": "it-session", "ttl_days": 1})
    assert resp.status_code == 201, resp.text
