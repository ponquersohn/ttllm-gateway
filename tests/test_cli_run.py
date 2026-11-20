"""`ttllm run claude`: token lifecycle, environment scrubbing, argument passthrough."""

from __future__ import annotations

import json

import httpx
import pytest
from typer.testing import CliRunner

from ttllm.cli import main, run

runner = CliRunner()

TOKEN_ID = "22222222-2222-2222-2222-222222222222"


class _Api:
    """Fake gateway recording the token calls."""

    def __init__(self, monkeypatch, delete_status: int = 204):
        self.requests: list[httpx.Request] = []
        self.delete_status = delete_status
        transport = httpx.MockTransport(self._handle)
        # A fresh client per get_client() call: each is closed by its `with` block.
        monkeypatch.setattr(
            run, "get_client", lambda: httpx.Client(base_url="http://test", transport=transport)
        )
        monkeypatch.setattr(run.TTLLMClient, "base_url_from_session", staticmethod(lambda: "https://gw.example/"))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST" and request.url.path == "/me/tokens":
            return httpx.Response(201, json={
                "access_token": "tok-123", "token_type": "bearer", "permissions": ["llm.invoke"],
                "id": TOKEN_ID, "label": "x", "expires_at": None,
            })
        if request.method == "DELETE":
            return httpx.Response(self.delete_status)
        return httpx.Response(500)

    def calls(self, method: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method]


class _Child:
    """Stand-in for subprocess.Popen capturing what was launched."""

    launched: list[dict] = []
    rc = 0
    raise_on_start: Exception | None = None

    def __init__(self, argv, env):
        if _Child.raise_on_start:
            raise _Child.raise_on_start
        _Child.launched.append({"argv": argv, "env": env})

    def wait(self):
        return _Child.rc

    def send_signal(self, signum):
        pass


@pytest.fixture
def api(monkeypatch):
    return _Api(monkeypatch)


@pytest.fixture(autouse=True)
def child(monkeypatch):
    _Child.launched = []
    _Child.rc = 0
    _Child.raise_on_start = None
    monkeypatch.setattr(run.subprocess, "Popen", _Child)
    monkeypatch.setattr(run.shutil, "which", lambda name: f"/usr/bin/{name}")
    return _Child


def _invoke(*args, env=None):
    return runner.invoke(main.app, ["run", "claude", *args], env=env)


def test_mints_runs_and_revokes(api, child):
    result = _invoke()
    assert result.exit_code == 0, result.output

    (post,) = api.calls("POST")
    body = json.loads(post.content)
    assert body["ttl_days"] == 1
    assert body["permissions"] == ["llm.invoke"]
    assert body["label"].startswith("ttllm run claude@")

    (delete,) = api.calls("DELETE")
    assert delete.url.path == f"/me/tokens/{TOKEN_ID}"

    env = child.launched[0]["env"]
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example/anthropic"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "tok-123"


@pytest.mark.parametrize(
    "args",
    [
        ["--resume"],
        ["-p", "hello", "--model", "x"],
        ["--model", "opus", "--verbose"],
        ["--", "--help"],
    ],
)
def test_args_pass_through(api, child, args):
    result = _invoke(*args)
    assert result.exit_code == 0, result.output
    expected = [a for a in args if a != "--"]
    assert child.launched[0]["argv"] == ["/usr/bin/claude", *expected]


def test_own_options_are_consumed(api, child):
    result = _invoke("--ttl-days", "3", "--small-model", "haiku-gw", "--continue")
    assert result.exit_code == 0, result.output
    assert json.loads(api.calls("POST")[0].content)["ttl_days"] == 3
    assert child.launched[0]["argv"] == ["/usr/bin/claude", "--continue"]
    assert child.launched[0]["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "haiku-gw"


def test_env_scrubbed(api, child):
    env = {
        "ANTHROPIC_API_KEY": "real-key",
        "ANTHROPIC_MODEL": "m",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CONFIG_DIR": "/cfg",
        "AWS_BEARER_TOKEN_BEDROCK": "b",
        "HOME_SOMETHING": "kept",
    }
    result = _invoke("--keep-env", "CLAUDE_CONFIG_DIR", env=env)
    assert result.exit_code == 0, result.output
    child_env = child.launched[0]["env"]
    for gone in ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "CLAUDE_CODE_USE_BEDROCK", "AWS_BEARER_TOKEN_BEDROCK"):
        assert gone not in child_env
    assert child_env["CLAUDE_CONFIG_DIR"] == "/cfg"
    assert child_env["HOME_SOMETHING"] == "kept"


def test_exit_code_propagated_and_token_revoked(api, child):
    child.rc = 7
    result = _invoke()
    assert result.exit_code == 7
    assert len(api.calls("DELETE")) == 1


def test_revoked_when_launch_fails(api, child):
    child.raise_on_start = OSError("exec format error")
    result = _invoke()
    assert result.exit_code != 0
    assert len(api.calls("DELETE")) == 1


def test_revoke_failure_warns(monkeypatch, child):
    api = _Api(monkeypatch, delete_status=500)
    result = _invoke()
    assert result.exit_code == 0
    assert "ttllm me tokens delete" in result.output
    assert len(api.calls("DELETE")) == 1


def test_missing_binary_mints_nothing(api, monkeypatch):
    monkeypatch.setattr(run.shutil, "which", lambda name: None)
    result = _invoke()
    assert result.exit_code == 1
    assert "claude not found" in result.output
    assert api.requests == []
