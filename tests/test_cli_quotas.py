"""CLI behaviour for quota management: the request bodies built, and argument rules."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from ttllm.cli import quotas, rules

runner = CliRunner()

QUOTA_ID = "11111111-1111-1111-1111-111111111111"
QUOTA = {
    "id": QUOTA_ID,
    "name": "daily",
    "description": None,
    "measure": "cost",
    "limit_value": "50",
    "dimensions": ["user"],
    "window_type": "calendar",
    "window_unit": "day",
    "window_duration_seconds": None,
    "timezone": "UTC",
    "enforcement": "soft",
    "on_exceeded": {},
    "enabled": True,
}


class _Recorder:
    """A real httpx client over a fake transport, recording what the CLI sent."""

    def __init__(self, monkeypatch):
        self.requests: list[httpx.Request] = []
        transport = httpx.MockTransport(self._handle)
        client = httpx.Client(base_url="http://test", transport=transport)
        for module in (quotas, rules):
            monkeypatch.setattr(module, "get_client", lambda: client)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/admin/quotas":
            return httpx.Response(200, json={"items": [QUOTA], "total": 1, "offset": 0, "limit": 200})
        if path == "/admin/rules":
            return httpx.Response(201, json={"id": "r", "name": "rule", "weight": 0})
        if path.endswith("/reset"):
            return httpx.Response(200, json={"buckets_reset": 3})
        return httpx.Response(200 if request.method == "PATCH" else 201, json=QUOTA)

    def body(self, method: str = "POST") -> dict:
        request = next(r for r in reversed(self.requests) if r.method == method)
        return json.loads(request.content)


@pytest.fixture
def api(monkeypatch):
    return _Recorder(monkeypatch)


def _invoke(app, *args):
    return runner.invoke(app, list(args))


class TestCreate:
    def test_calendar_defaults_to_per_user(self, api):
        result = _invoke(quotas.app, "create", "--name", "daily", "--measure", "cost", "--limit", "50", "--window", "day")
        assert result.exit_code == 0, result.output
        body = api.body()
        assert body["window_type"] == "calendar"
        assert body["window_unit"] == "day"
        assert body["window_duration_seconds"] is None
        assert body["dimensions"] == ["user"]
        assert body["limit_value"] == "50"

    def test_session_window(self, api):
        result = _invoke(
            quotas.app, "create", "--name", "s", "--measure", "tokens", "--limit", "1000", "--session-seconds", "10800"
        )
        assert result.exit_code == 0, result.output
        body = api.body()
        assert body["window_type"] == "session"
        assert body["window_duration_seconds"] == 10800
        assert body["window_unit"] is None

    def test_repeatable_dimensions(self, api):
        result = _invoke(
            quotas.app,
            "create", "--name", "d", "--measure", "requests", "--limit", "5", "--window", "hour",
            "--dimension", "user", "--dimension", "model",
        )
        assert result.exit_code == 0, result.output
        assert api.body()["dimensions"] == ["user", "model"]

    def test_shared_has_no_dimensions(self, api):
        result = _invoke(
            quotas.app, "create", "--name", "d", "--measure", "cost", "--limit", "5", "--window", "month", "--shared"
        )
        assert result.exit_code == 0, result.output
        assert api.body()["dimensions"] == []

    @pytest.mark.parametrize(
        "extra",
        [
            [],
            ["--window", "day", "--session-seconds", "60"],
            ["--window", "day", "--shared", "--dimension", "user"],
        ],
    )
    def test_rejects_conflicting_arguments(self, api, extra):
        result = _invoke(quotas.app, "create", "--name", "d", "--measure", "cost", "--limit", "5", *extra)
        assert result.exit_code == 1
        assert not [r for r in api.requests if r.method == "POST"]


class TestOtherCommands:
    def test_update_resolves_name_and_sends_only_given_fields(self, api):
        result = _invoke(quotas.app, "update", "daily", "--limit", "75")
        assert result.exit_code == 0, result.output
        patch = next(r for r in api.requests if r.method == "PATCH")
        assert patch.url.path == f"/admin/quotas/{QUOTA_ID}"
        assert json.loads(patch.content) == {"limit_value": "75"}

    def test_update_with_nothing_to_do(self, api):
        assert _invoke(quotas.app, "update", "daily").exit_code == 1

    def test_unknown_name_fails(self, api):
        assert _invoke(quotas.app, "update", "nope", "--limit", "1").exit_code == 1

    def test_reset(self, api):
        result = _invoke(quotas.app, "reset", "daily")
        assert result.exit_code == 0, result.output
        assert "3 bucket(s)" in result.output


class TestRulesApplyQuota:
    CONDITIONS = '{"logic":"and","conditions":[]}'

    def test_resolves_names_to_ids(self, api):
        result = _invoke(
            rules.app, "create", "--name", "r", "--conditions", self.CONDITIONS, "--apply-quota", "daily"
        )
        assert result.exit_code == 0, result.output
        assert api.body()["action"] == {"type": "apply_quota", "quota_ids": [QUOTA_ID]}

    @pytest.mark.parametrize("extra", [[], ["--action", '{"type":"allow"}', "--apply-quota", "daily"]])
    def test_create_needs_exactly_one_action(self, api, extra):
        result = _invoke(rules.app, "create", "--name", "r", "--conditions", self.CONDITIONS, *extra)
        assert result.exit_code == 1

    def test_plain_action_still_works(self, api):
        result = _invoke(
            rules.app, "create", "--name", "r", "--conditions", self.CONDITIONS, "--action", '{"type":"allow"}'
        )
        assert result.exit_code == 0, result.output
        assert api.body()["action"] == {"type": "allow"}


OVERRIDE_ID = "22222222-2222-2222-2222-222222222222"
ISSUED = {
    "id": OVERRIDE_ID,
    "code": "ttqo_secretcode",
    "reason": "release crunch",
    "override_duration_seconds": 14400,
    "code_expires_at": "2026-11-21T12:00:00Z",
}
LISTED = {
    "id": OVERRIDE_ID,
    "state": "active",
    "reason": "release crunch",
    "created_by_email": "admin@example.com",
    "override_duration_seconds": 14400,
    "code_expires_at": "2026-11-21T12:00:00Z",
    "redeemed_by_email": "dev@example.com",
    "redeemed_at": "2026-11-20T12:00:00Z",
    "override_expires_at": "2026-11-20T16:00:00Z",
    "revoked_at": None,
    "created_at": "2026-11-20T11:00:00Z",
}


class _OverrideApi:
    def __init__(self, monkeypatch):
        self.requests: list[httpx.Request] = []
        client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(self._handle))
        monkeypatch.setattr(quotas, "get_client", lambda: client)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            return httpx.Response(201, json=ISSUED)
        if request.method == "GET":
            return httpx.Response(200, json={"items": [LISTED], "total": 1, "offset": 0, "limit": 50})
        return httpx.Response(204)


@pytest.fixture
def override_api(monkeypatch):
    return _OverrideApi(monkeypatch)


class TestOverrides:
    def test_issue_parses_durations_and_shows_code(self, override_api):
        result = _invoke(
            quotas.app, "overrides", "issue", "--duration", "4h", "--code-ttl", "2d", "--reason", "release crunch"
        )
        assert result.exit_code == 0, result.output
        request = override_api.requests[-1]
        assert request.url.path == "/admin/quota-overrides"
        assert json.loads(request.content) == {
            "override_duration_seconds": 14400,
            "code_ttl_seconds": 172800,
            "reason": "release crunch",
        }
        assert "ttqo_secretcode" in result.output

    def test_issue_requires_reason(self, override_api):
        result = _invoke(quotas.app, "overrides", "issue", "--duration", "4h")
        assert result.exit_code != 0
        assert not override_api.requests

    def test_issue_rejects_bad_duration(self, override_api):
        result = _invoke(quotas.app, "overrides", "issue", "--duration", "soon", "--reason", "x y z")
        assert result.exit_code != 0
        assert not override_api.requests

    def test_list_with_state(self, override_api):
        result = runner.invoke(quotas.app, ["overrides", "list", "--state", "active"], env={"COLUMNS": "250"})
        assert result.exit_code == 0, result.output
        assert override_api.requests[-1].url.params["state"] == "active"
        assert "dev@example.com" in result.output

    def test_list_rejects_unknown_state(self, override_api):
        assert _invoke(quotas.app, "overrides", "list", "--state", "bogus").exit_code != 0
        assert not override_api.requests

    def test_revoke_with_reason(self, override_api):
        result = _invoke(quotas.app, "overrides", "revoke", OVERRIDE_ID, "--reason", "done early")
        assert result.exit_code == 0, result.output
        request = override_api.requests[-1]
        assert request.method == "DELETE"
        assert request.url.path == f"/admin/quota-overrides/{OVERRIDE_ID}"
        assert request.url.params["reason"] == "done early"
