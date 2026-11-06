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
