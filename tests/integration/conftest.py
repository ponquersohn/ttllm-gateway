"""Fixtures for end-to-end integration tests against a running gateway.

These tests require the docker-compose.integration.yml stack to be up. They are skipped
unless run with `-m integration`. The gateway base URL defaults to http://localhost:8000
and can be overridden with TTLLM_TEST_BASE_URL.

Bootstrap relies on the admin user seeded by migration 006 (admin@localhost / admin,
overridable via TTLLM_ADMIN_PASSWORD).
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

BASE_URL = os.environ.get("TTLLM_TEST_BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = "admin@localhost"
ADMIN_PASSWORD = os.environ.get("TTLLM_ADMIN_PASSWORD", "admin")
# Compose-internal URL the gateway uses to reach the fake Bedrock (NOT localhost).
FAKE_BEDROCK_URL = os.environ.get("TTLLM_TEST_BEDROCK_URL", "http://fake-bedrock:9099")


def pytest_collection_modifyitems(items):
    """Mark every test under tests/integration/ as `integration` automatically."""
    for item in items:
        if "tests/integration/" in item.nodeid or "tests\\integration\\" in item.nodeid:
            item.add_marker(pytest.mark.integration)


def _wait_for_gateway(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.0)
    raise RuntimeError(f"Gateway at {url} not healthy within {timeout}s: {last_err}")


@pytest.fixture(scope="session", autouse=True)
def _gateway_ready() -> None:
    _wait_for_gateway(BASE_URL)


@pytest.fixture(scope="session")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as c:
        yield c


@pytest.fixture(scope="session")
def admin_token(client: httpx.Client) -> str:
    resp = client.post("/auth/token", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def admin_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


def _unique(prefix: str) -> str:
    # Per-run uniqueness without Date/random: use the pid + monotonic ns.
    return f"{prefix}-{os.getpid()}-{time.monotonic_ns()}"


@pytest.fixture(scope="session")
def bedrock_model(client: httpx.Client, admin_headers: dict[str, str]) -> dict:
    """Create a Bedrock model pointed at the fake Bedrock server."""
    name = _unique("it-model")
    resp = client.post(
        "/admin/models",
        headers=admin_headers,
        json={
            "name": name,
            "provider": "bedrock",
            "provider_model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
            "config_json": {
                "region": "us-east-1",
                "endpoint_url": FAKE_BEDROCK_URL,
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            },
            "input_cost_per_1k": "0.003",
            "output_cost_per_1k": "0.015",
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest.fixture(scope="session")
def gateway_user_token(client: httpx.Client, admin_headers: dict[str, str], bedrock_model: dict) -> str:
    """Create a user, assign the model, and mint a gateway (llm.invoke) token."""
    email = f"{_unique('it-user')}@example.com"
    u = client.post(
        "/admin/users",
        headers=admin_headers,
        json={"name": "Integration User", "email": email, "password": "Integration1!"},
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
        f"/admin/models/{bedrock_model['id']}/assign",
        headers=admin_headers,
        json={"user_ids": [user_id]},
    )
    assert a.status_code == 201, a.text

    t = client.post(
        "/admin/tokens",
        headers=admin_headers,
        json={"user_id": user_id, "label": "integration", "permissions": ["llm.invoke"]},
    )
    assert t.status_code == 201, t.text
    return t.json()["access_token"]
