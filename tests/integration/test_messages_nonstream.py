"""Non-streaming /v1/messages against the real boto3 → fake Bedrock path."""

from __future__ import annotations

import os
import time

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


def test_nonstreaming_audit_cost_and_metadata(
    client: httpx.Client, admin_headers: dict, gateway_user_token: str, bedrock_model: dict
):
    """A completed request records an authoritative total_cost and a provider_metadata blob,
    and the per-model cost breakdown sums the stored totals."""
    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": gateway_user_token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "cost check"}],
            "max_tokens": 128,
        },
    )
    assert resp.status_code == 200, resp.text

    # Audit row carries the provider-computed cost + metadata blob.
    logs = client.get(
        "/admin/audit-logs",
        headers=admin_headers,
        params={"model_id": bedrock_model["id"], "limit": 50},
    )
    assert logs.status_code == 200, logs.text
    items = logs.json()["items"]
    assert items, "expected at least one audit row"
    latest = items[0]
    assert latest["total_cost"] is not None
    assert latest["provider_metadata"]["provider"] == "bedrock"
    assert "cost" in latest["provider_metadata"]
    assert "raw" in latest["provider_metadata"]

    # Cost breakdown sums the stored totals for this model.
    costs = client.get(
        "/admin/usage/costs",
        headers=admin_headers,
        params={"model_id": bedrock_model["id"]},
    )
    assert costs.status_code == 200, costs.text
    rows = [r for r in costs.json() if r["model_name"] == bedrock_model["name"]]
    assert rows and float(rows[0]["total_cost"]) >= 0

    # Usage summary now reports a total_cost field.
    summary = client.get(
        "/admin/usage", headers=admin_headers, params={"model_id": bedrock_model["id"]}
    )
    assert summary.status_code == 200, summary.text
    assert "total_cost" in summary.json()


def test_usage_filter_by_email_and_by_user(
    client: httpx.Client, admin_headers: dict, bedrock_model: dict
):
    """Usage endpoints can scope by user email, and by-user lists spenders highest-cost first."""
    email = f"usage-email-probe-{os.getpid()}-{time.monotonic_ns()}@example.com"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Usage Probe", "email": email, "password": "Integration1!"},
    )
    assert u.status_code == 201, u.text
    user_id = u.json()["id"]

    client.post(
        f"/admin/users/{user_id}/permissions",
        headers=admin_headers,
        json={"permissions": ["llm.invoke"]},
    )
    client.post(
        f"/admin/models/{bedrock_model['id']}/assign",
        headers=admin_headers,
        json={"user_ids": [user_id]},
    )
    t = client.post(
        "/admin/tokens",
        headers=admin_headers,
        json={"user_id": user_id, "label": "usage-probe", "permissions": ["llm.invoke"]},
    )
    token = t.json()["access_token"]

    resp = client.post(
        "/anthropic/v1/messages",
        headers={"x-api-key": token},
        json={
            "model": bedrock_model["name"],
            "messages": [{"role": "user", "content": "email scoped"}],
            "max_tokens": 128,
        },
    )
    assert resp.status_code == 200, resp.text

    # Filtering the summary by email matches filtering by the resolved user id.
    by_email = client.get("/admin/usage", headers=admin_headers, params={"email": email})
    assert by_email.status_code == 200, by_email.text
    by_id = client.get("/admin/usage", headers=admin_headers, params={"user_id": user_id})
    assert by_email.json()["total_requests"] == by_id.json()["total_requests"] >= 1

    # Unknown email is a 404, not a silent unscoped query.
    missing = client.get(
        "/admin/usage", headers=admin_headers, params={"email": "nobody@example.com"}
    )
    assert missing.status_code == 404, missing.text

    # by-user lists the probe user, ordered by cost descending.
    by_user = client.get("/admin/usage/by-user", headers=admin_headers)
    assert by_user.status_code == 200, by_user.text
    rows = by_user.json()
    assert any(r["user_email"] == email for r in rows)
    costs = [float(r["total_cost"]) for r in rows]
    assert costs == sorted(costs, reverse=True)
