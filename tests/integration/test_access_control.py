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
