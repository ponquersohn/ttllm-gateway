"""End-to-end tests for the device code login flow (RFC 8628)."""

from __future__ import annotations

import httpx

from tests.integration.conftest import _unique


def _start(client: httpx.Client) -> dict:
    resp = client.post("/auth/device/code", json={"client_name": "it-device"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["interval"] == 5 and data["expires_in"] > 0
    assert data["user_code"] in data["verification_uri_complete"]
    return data


def _poll(client: httpx.Client, device_code: str) -> httpx.Response:
    return client.post("/auth/device/token", json={"device_code": device_code})


def _err(resp: httpx.Response) -> str:
    return resp.json()["error"]["type"]


def _make_user(client: httpx.Client, admin_headers: dict) -> tuple[str, dict]:
    email = f"{_unique('it-device-user')}@example.com"
    password = "Integration1!"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Device User", "email": email, "password": password},
    )
    assert u.status_code == 201, u.text
    login = client.post("/auth/token", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    return u.json()["id"], {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_approve_flow_issues_single_use_tokens(client: httpx.Client, admin_headers: dict):
    data = _start(client)

    pending = _poll(client, data["device_code"])
    assert pending.status_code == 400 and _err(pending) == "authorization_pending"

    lookup = client.get("/auth/device/lookup", params={"user_code": data["user_code"]}, headers=admin_headers)
    assert lookup.status_code == 200, lookup.text
    assert lookup.json()["client_name"] == "it-device"

    # Lowercase / no dash must still match.
    sloppy = data["user_code"].replace("-", "").lower()
    approve = client.post("/auth/device/approve", headers=admin_headers, json={"user_code": sloppy})
    assert approve.status_code == 204, approve.text

    tokens = _poll(client, data["device_code"])
    assert tokens.status_code == 200, tokens.text
    me = client.get("/me", headers={"Authorization": f"Bearer {tokens.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "admin@localhost"

    again = _poll(client, data["device_code"])
    assert again.status_code == 400 and _err(again) == "expired_token"

    # Code can't be approved twice either.
    reapprove = client.post("/auth/device/approve", headers=admin_headers, json={"user_code": data["user_code"]})
    assert reapprove.status_code == 404


def test_deny(client: httpx.Client, admin_headers: dict):
    data = _start(client)
    resp = client.post(
        "/auth/device/approve", headers=admin_headers, json={"user_code": data["user_code"], "approve": False}
    )
    assert resp.status_code == 204
    denied = _poll(client, data["device_code"])
    assert denied.status_code == 400 and _err(denied) == "access_denied"


def test_slow_down(client: httpx.Client):
    data = _start(client)
    assert _err(_poll(client, data["device_code"])) == "authorization_pending"
    assert _err(_poll(client, data["device_code"])) == "slow_down"


def test_unknown_device_code(client: httpx.Client):
    resp = _poll(client, "not-a-real-code")
    assert resp.status_code == 400 and _err(resp) == "expired_token"


def test_approval_requires_authentication(client: httpx.Client):
    data = _start(client)
    resp = client.post("/auth/device/approve", json={"user_code": data["user_code"]})
    assert resp.status_code == 401


def test_gateway_api_token_cannot_approve(client: httpx.Client, gateway_user_token: str):
    data = _start(client)
    headers = {"Authorization": f"Bearer {gateway_user_token}"}
    assert client.post("/auth/device/approve", headers=headers, json={"user_code": data["user_code"]}).status_code == 403
    assert client.get("/auth/device/lookup", headers=headers, params={"user_code": data["user_code"]}).status_code == 403


def test_deactivated_user_gets_no_tokens(client: httpx.Client, admin_headers: dict):
    user_id, user_headers = _make_user(client, admin_headers)
    data = _start(client)
    approve = client.post("/auth/device/approve", headers=user_headers, json={"user_code": data["user_code"]})
    assert approve.status_code == 204, approve.text

    deact = client.patch(f"/admin/users/{user_id}", headers=admin_headers, json={"is_active": False})
    assert deact.status_code == 200, deact.text

    resp = _poll(client, data["device_code"])
    assert resp.status_code == 403 and _err(resp) == "permission_error"
