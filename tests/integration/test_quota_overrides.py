"""Quota override codes end to end: issue, preview, redeem, bypass, revoke, and audit."""

from __future__ import annotations

from tests.integration.test_quota_rules import _new_user, _quotas, _send, _unique

PASSWORD = "Integration1!"


def _login(client, email: str) -> dict:
    resp = client.post("/auth/token", json={"email": email, "password": PASSWORD})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _issue(client, admin_headers, **overrides) -> dict:
    body = {"override_duration_seconds": 600, "code_ttl_seconds": 600, "reason": "integration test", **overrides}
    resp = client.post("/admin/quota-overrides", headers=admin_headers, json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_reason_is_required(client, admin_headers):
    for reason in ("", "  ", "x"):
        resp = client.post(
            "/admin/quota-overrides",
            headers=admin_headers,
            json={"override_duration_seconds": 600, "code_ttl_seconds": 600, "reason": reason},
        )
        assert resp.status_code == 422, resp.text
    missing = client.post(
        "/admin/quota-overrides", headers=admin_headers, json={"override_duration_seconds": 600, "code_ttl_seconds": 600}
    )
    assert missing.status_code == 422


def test_override_bypasses_quota_until_revoked(client, admin_headers, bedrock_model):
    api_token, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    session = _login(client, email)
    with _quotas(client, admin_headers, email, {"limit_value": "1"}) as (quota,):
        assert _send(client, api_token, bedrock_model["name"]).status_code == 200
        assert _send(client, api_token, bedrock_model["name"]).status_code == 429

        issued = _issue(client, admin_headers, reason="unblock integration user")
        code = issued["code"]
        assert code.startswith("ttqo_")

        # An API token can't redeem (or preview) a code: login sessions only.
        api_headers = {"Authorization": f"Bearer {api_token}"}
        assert client.post("/me/quota-overrides/redeem", headers=api_headers, json={"code": code}).status_code == 403

        preview = client.post("/me/quota-overrides/lookup", headers=session, json={"code": code})
        assert preview.status_code == 200, preview.text
        assert preview.json()["reason"] == "unblock integration user"
        assert preview.json()["override_duration_seconds"] == 600

        redeemed = client.post("/me/quota-overrides/redeem", headers=session, json={"code": code})
        assert redeemed.status_code == 200, redeemed.text
        # Single use.
        again = client.post("/me/quota-overrides/redeem", headers=session, json={"code": code})
        assert again.status_code == 404
        assert client.get("/me/quota-overrides", headers=session).json()[0]["id"] == issued["id"]

        buckets_before = client.get(f"/admin/quotas/{quota['id']}/buckets", headers=admin_headers).json()["items"]
        for _ in range(3):
            ok = _send(client, api_token, bedrock_model["name"])
            assert ok.status_code == 200, ok.text
        buckets_after = client.get(f"/admin/quotas/{quota['id']}/buckets", headers=admin_headers).json()["items"]
        assert [b["used"] for b in buckets_after] == [b["used"] for b in buckets_before]

        request_id = ok.headers["x-request-id"]
        logs = client.get("/admin/audit-logs", headers=admin_headers, params={"request_id": request_id})
        assert logs.status_code == 200, logs.text
        entries = logs.json()["items"] if isinstance(logs.json(), dict) else logs.json()
        assert entries[0]["metadata_json"]["quota_override"]["reason"] == "unblock integration user"

        listed = client.get("/admin/quota-overrides", headers=admin_headers, params={"state": "active"}).json()
        assert issued["id"] in [o["id"] for o in listed["items"]]

        revoke = client.delete(f"/admin/quota-overrides/{issued['id']}", headers=admin_headers)
        assert revoke.status_code == 204
        assert client.delete(f"/admin/quota-overrides/{issued['id']}", headers=admin_headers).status_code == 404
        assert _send(client, api_token, bedrock_model["name"]).status_code == 429
        assert client.get("/me/quota-overrides", headers=session).json() == []


def test_issuer_can_redeem_own_code_and_revoked_codes_fail(client, admin_headers):
    own = _issue(client, admin_headers)
    resp = client.post("/me/quota-overrides/redeem", headers=admin_headers, json={"code": own["code"]})
    assert resp.status_code == 200, resp.text
    client.delete(f"/admin/quota-overrides/{own['id']}", headers=admin_headers)

    dead = _issue(client, admin_headers)
    client.delete(f"/admin/quota-overrides/{dead['id']}", headers=admin_headers)
    resp = client.post("/me/quota-overrides/lookup", headers=admin_headers, json={"code": dead["code"]})
    assert resp.status_code == 404


def test_issuing_requires_permission(client, admin_headers, bedrock_model):
    _, email, _ = _new_user(client, admin_headers, [bedrock_model["id"]])
    session = _login(client, email)
    resp = client.post(
        "/admin/quota-overrides",
        headers=session,
        json={"override_duration_seconds": 600, "code_ttl_seconds": 600, "reason": _unique("nope")},
    )
    assert resp.status_code == 403
    assert client.get("/admin/quota-overrides", headers=session).status_code == 403
