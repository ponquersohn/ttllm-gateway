"""Tests for the universal --json output mechanism (TtllmTyper + json_mode)."""

from __future__ import annotations

import contextlib
import json
import re

from typer.testing import CliRunner

from ttllm.cli import _common, users

runner = CliRunner()

# Rich may colorize help output with ANSI escapes (e.g. in CI where color is
# forced), which splits "--json" across escape sequences. Strip them so the
# substring assertion does not depend on terminal color detection.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --- The TtllmTyper mechanism itself ---


def test_ttllmtyper_injects_json_flag_and_sets_mode():
    app = _common.TtllmTyper()

    @app.command("ping")
    def ping():
        if _common.json_mode():
            _common.print_json({"pong": True})
        else:
            _common.console.print("pong")

    # A second command keeps Typer from collapsing the single command into the
    # app callback (which would make the subcommand name an unexpected argument).
    @app.command("noop")
    def noop():
        pass

    # --json shows up in help even though `ping` declares no parameter
    help_out = _strip_ansi(runner.invoke(app, ["ping", "--help"]).output)
    assert "--json" in help_out

    plain = runner.invoke(app, ["ping"])
    assert plain.output.strip() == "pong"

    as_json = runner.invoke(app, ["ping", "--json"])
    assert json.loads(as_json.output) == {"pong": True}


def test_json_mode_defaults_false_outside_command():
    assert _common.json_mode() is False


# --- A mutation command honours --json end to end ---


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    @contextlib.contextmanager
    def _cm(self):
        yield self

    def post(self, *args, **kwargs):
        return _FakeResponse(200, self._payload)

    def delete(self, *args, **kwargs):
        return _FakeResponse(204)


def _patch_client(monkeypatch, module, payload):
    client = _FakeClient(payload)
    monkeypatch.setattr(module, "get_client", lambda: client._cm())
    return client


def test_users_create_json(monkeypatch):
    payload = {"id": "abc-123", "name": "Alice", "email": "alice@example.com"}
    _patch_client(monkeypatch, users, payload)

    result = runner.invoke(
        users.app,
        ["create", "--name", "Alice", "--email", "alice@example.com", "--json"],
    )
    assert result.exit_code == 0
    assert json.loads(result.output) == payload


def test_users_create_plain_unchanged(monkeypatch):
    payload = {"id": "abc-123", "name": "Alice", "email": "alice@example.com"}
    _patch_client(monkeypatch, users, payload)

    result = runner.invoke(
        users.app,
        ["create", "--name", "Alice", "--email", "alice@example.com"],
    )
    assert result.exit_code == 0
    assert "User created:" in result.output
    assert "abc-123" in result.output


def test_users_delete_json_emits_synthetic_status(monkeypatch):
    _patch_client(monkeypatch, users, {})

    result = runner.invoke(users.app, ["delete", "some-id", "--use-ids", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"status": "deactivated", "id": "some-id"}
