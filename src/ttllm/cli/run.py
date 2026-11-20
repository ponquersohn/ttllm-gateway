"""`ttllm run <tool>`: launch a local tool against the gateway with a throwaway token.

The token is minted from the login session (`POST /me/tokens`), handed to the
child process through its environment, and revoked when the child exits.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
from typing import Optional

import typer

from ttllm.cli._common import console, get_client, handle_response
from ttllm.cli.client import TTLLMClient

# Plain Typer (not TtllmTyper): an injected --json flag would swallow the
# passthrough argument of the same name.
app = typer.Typer(help="Run a local tool through the gateway with a temporary token")

# Credentials that would make Claude Code bypass the gateway, on top of the
# ANTHROPIC_* / CLAUDE* prefixes.
_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE")
_STRIP_EXTRA = {"AWS_BEARER_TOKEN_BEDROCK"}

_PASSTHROUGH = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "allow_interspersed_args": False,
}


@app.callback()
def _run():
    """Run a local tool through the gateway with a temporary token."""


def build_claude_env(
    base_env: dict[str, str],
    *,
    base_url: str,
    token: str,
    keep: list[str],
    small_model: str | None,
) -> dict[str, str]:
    """Environment for Claude Code: scrubbed of conflicting vars, pointed at the gateway."""
    keep_set = set(keep)
    env = {
        k: v
        for k, v in base_env.items()
        if k in keep_set or not (k.startswith(_STRIP_PREFIXES) or k in _STRIP_EXTRA)
    }
    env["ANTHROPIC_BASE_URL"] = f"{base_url.rstrip('/')}/anthropic"
    # Sent as `Authorization: Bearer`, the only scheme the gateway accepts
    # (ANTHROPIC_API_KEY would be sent as x-api-key).
    env["ANTHROPIC_AUTH_TOKEN"] = token
    if small_model:
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = small_model
    return env


def _run_child(argv: list[str], env: dict[str, str]) -> int:
    """Run the child in the foreground, letting it own Ctrl-C; forward TERM/HUP."""
    proc = subprocess.Popen(argv, env=env)

    def _forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass

    previous = {
        signal.SIGINT: signal.signal(signal.SIGINT, signal.SIG_IGN),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _forward),
        signal.SIGHUP: signal.signal(signal.SIGHUP, _forward),
    }
    try:
        return proc.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _revoke(token_id: str) -> None:
    try:
        with get_client() as client:
            resp = client.delete(f"/me/tokens/{token_id}")
        if resp.status_code in (204, 404):
            return
        problem = f"HTTP {resp.status_code}"
    except Exception as e:  # network down, session gone, ...
        problem = str(e) or type(e).__name__
    console.print(
        f"[yellow]Could not revoke temporary token {token_id} ({problem}). "
        f"Revoke it with: ttllm me tokens delete {token_id}[/yellow]"
    )


@app.command("claude", context_settings=_PASSTHROUGH)
def claude(
    ctx: typer.Context,
    ttl_days: int = typer.Option(1, "--ttl-days", min=1, help="Lifetime of the temporary token (revoked on exit anyway)"),
    small_model: Optional[str] = typer.Option(
        None, "--small-model", help="Gateway model for Claude Code's background/haiku calls (ANTHROPIC_DEFAULT_HAIKU_MODEL)"
    ),
    keep_env: list[str] = typer.Option([], "--keep-env", help="ANTHROPIC_*/CLAUDE* variable to keep (repeatable)"),
    claude_bin: Optional[str] = typer.Option(None, "--claude-bin", help="Path to the claude executable"),
):
    """Run Claude Code through the gateway. Extra arguments are passed to claude
    (use `--` before claude flags that clash with these, e.g. `-- --help`)."""
    binary = claude_bin or shutil.which("claude")
    if not binary:
        console.print("[red]claude not found on PATH. Install Claude Code or pass --claude-bin.[/red]")
        raise typer.Exit(1)

    with get_client() as client:
        created = handle_response(
            client.post(
                "/me/tokens",
                json={
                    "label": f"ttllm run claude@{socket.gethostname()}",
                    "ttl_days": ttl_days,
                    "permissions": ["llm.invoke"],
                },
            )
        )

    try:
        env = build_claude_env(
            dict(os.environ),
            base_url=TTLLMClient.base_url_from_session(),
            token=created["access_token"],
            keep=keep_env,
            small_model=small_model,
        )
        rc = _run_child([binary, *ctx.args], env)
    finally:
        _revoke(created["id"])

    raise typer.Exit(rc)
