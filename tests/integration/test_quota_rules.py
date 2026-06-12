"""End-to-end moving-window quota rule: throttle a user after N requests.

Uses the `requests` measure for a deterministic threshold (cost/token totals
depend on fake-Bedrock token math; request count is exact). Creates a quota rule
via the admin API, fires requests until the window threshold trips, and asserts a
429 with a rendered message and a Retry-After header. The rule is global, so it is
torn down in a finally block to avoid throttling other integration tests.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest


def _fresh_gateway_token(client: httpx.Client, admin_headers: dict, model_id: str) -> str:
    """Mint a brand-new user + gateway token so the quota window starts empty.

    The session-scoped `gateway_user_token` accumulates audit rows across the
    other message tests; a per-test user isolates the moving-window count.
    """
    uniq = f"{os.getpid()}-{time.monotonic_ns()}"
    email = f"quota-user-{uniq}@example.com"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Quota User", "email": email, "password": "Integration1!"},
    )
    assert u.status_code == 201, u.text
    user_id = u.json()["id"]

    p = client.post(
        f"/admin/users/{user_id}/permissions",
        headers=admin_headers,
        json={"permissions": ["llm.invoke"]},
    )
    assert p.status_code == 201, p.text

    a = client.post(
        f"/admin/models/{model_id}/assign",
        headers=admin_headers,
        json={"user_ids": [user_id]},
    )
    assert a.status_code == 201, a.text

    t = client.post(
        "/admin/tokens",
        headers=admin_headers,
        json={"user_id": user_id, "label": "quota-test", "permissions": ["llm.invoke"]},
    )
    assert t.status_code == 201, t.text
    return t.json()["access_token"]


def _send(client: httpx.Client, token: str, model_name: str) -> httpx.Response:
    return client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": token},
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": "quota check"}],
            "max_tokens": 64,
        },
    )


def test_request_quota_throttles_with_429_and_retry_after(
    client: httpx.Client,
    admin_headers: dict,
    bedrock_model: dict,
):
    # Fresh user so the moving window starts empty (session token is shared).
    token = _fresh_gateway_token(client, admin_headers, bedrock_model["id"])

    # A short-window rule: more than 2 successful requests in the last hour → 429.
    # window=3600 so contributing rows don't age out mid-test.
    rule_resp = client.post(
        "/admin/rules",
        headers=admin_headers,
        json={
            "name": "it-request-quota",
            "weight": 1000,
            "conditions": {
                "logic": "and",
                "conditions": [
                    {"type": "quota", "field": "requests", "operator": "gt",
                     "value": 2, "window": 3600},
                ],
            },
            "action": {
                "type": "block",
                "status_code": 429,
                "message": "Rate limit: {{quota.requests.value}} reqs. Retry in {{quota.requests.next_free}}s.",
            },
        },
    )
    assert rule_resp.status_code == 201, rule_resp.text
    rule_id = rule_resp.json()["id"]

    try:
        # Three successful requests build up the window (each writes a 200 audit row).
        for _ in range(3):
            ok = _send(client, token, bedrock_model["name"])
            assert ok.status_code == 200, ok.text

        # The next request now exceeds the threshold and is throttled.
        blocked = _send(client, token, bedrock_model["name"])
        assert blocked.status_code == 429, blocked.text

        # The gateway wraps HTTPException detail as {"type":"error","error":{...}}.
        body = blocked.json()
        message = body.get("error", body.get("detail", {})).get("message", "")
        assert "Rate limit" in message, body
        # The {{quota.requests.value}} token rendered to a real number.
        assert "{{" not in message

        retry_after = blocked.headers.get("Retry-After")
        assert retry_after is not None and int(retry_after) > 0
    finally:
        client.delete(f"/admin/rules/{rule_id}", headers=admin_headers)


def test_no_quota_rule_allows_request(
    client: httpx.Client,
    gateway_user_token: str,
    bedrock_model: dict,
):
    # With the throttle rule removed (previous test cleaned up), requests pass.
    resp = _send(client, gateway_user_token, bedrock_model["name"])
    assert resp.status_code == 200, resp.text
