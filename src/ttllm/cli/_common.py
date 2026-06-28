"""Shared CLI utilities: console, client helpers, name resolution."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json

import httpx
import typer
from rich.console import Console

from ttllm.cli.client import TTLLMClient

console = Console()

JSON_OPTION = typer.Option(False, "--json", help="Output raw JSON")

_json_mode: contextvars.ContextVar[bool] = contextvars.ContextVar("json_mode", default=False)


def json_mode() -> bool:
    """Return True if the current command was invoked with --json."""
    return _json_mode.get()


def _inject_json(decorator, fn):
    """Append a hidden --json option to a command/callback and expose it via json_mode().

    Typer builds CLI flags from the function signature, so we inject a private
    keyword-only parameter (bound to JSON_OPTION) into a wrapper's signature.
    The wrapper stashes the value in a ContextVar that json_mode() reads, so
    commands never need to declare a --json parameter themselves.
    """
    sig = inspect.signature(fn)
    # Already wrapped (e.g. a function with both @command and @callback stacked):
    # register as-is rather than injecting a duplicate parameter.
    if "_json_out" in sig.parameters:
        return decorator(fn)
    params = list(sig.parameters.values()) + [
        inspect.Parameter(
            "_json_out",
            inspect.Parameter.KEYWORD_ONLY,
            default=JSON_OPTION,
            annotation=bool,
        )
    ]

    @functools.wraps(fn)
    def inner(*args, _json_out=False, **kwargs):
        token = _json_mode.set(_json_out)
        try:
            return fn(*args, **kwargs)
        finally:
            _json_mode.reset(token)

    inner.__signature__ = sig.replace(parameters=params)
    return decorator(inner)


class TtllmTyper(typer.Typer):
    """Typer subclass that gives every command (and callback) a --json flag."""

    def command(self, *args, **kwargs):
        decorator = super().command(*args, **kwargs)
        return lambda fn: _inject_json(decorator, fn)

    def callback(self, *args, **kwargs):
        decorator = super().callback(*args, **kwargs)
        return lambda fn: _inject_json(decorator, fn)


def print_json(data) -> None:
    """Print data as formatted JSON and exit.

    soft_wrap avoids Rich inserting line breaks into long string values (e.g.
    JWT tokens), which would otherwise corrupt the JSON when piped to a parser.
    """
    console.print(json.dumps(data, indent=2, default=str), highlight=False, soft_wrap=True)


def get_client() -> TTLLMClient:
    session = TTLLMClient.load_session()
    if not session or not session.get("access_token"):
        console.print("[red]Not logged in. Run 'ttllm login' first.[/red]")
        raise typer.Exit(1)
    return TTLLMClient.from_session()


def handle_response(resp: httpx.Response) -> dict:
    if resp.status_code == 401:
        console.print("[red]Session expired. Run 'ttllm login' again.[/red]")
        raise typer.Exit(1)
    if resp.status_code >= 400:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")
        raise typer.Exit(1)
    return resp.json()


# --- Name resolution helpers ---


def resolve_user(client: httpx.Client, name: str) -> str:
    """Resolve a user name to a user ID."""
    data = handle_response(client.get("/admin/users", params={"limit": 200}))
    needle = name.lower()
    for u in data["items"]:
        if u["name"].lower() == needle or u["email"].lower() == needle:
            return u["id"]
    console.print(f"[red]User not found: {name}[/red]")
    raise typer.Exit(1)


def resolve_group(client: httpx.Client, name: str) -> str:
    """Resolve a group name to a group ID."""
    data = handle_response(client.get("/admin/groups", params={"limit": 200}))
    needle = name.lower()
    for g in data["items"]:
        if g["name"].lower() == needle:
            return g["id"]
    console.print(f"[red]Group not found: {name}[/red]")
    raise typer.Exit(1)


def resolve_model(client: httpx.Client, name: str) -> str:
    """Resolve a model name to a model ID."""
    data = handle_response(client.get("/admin/models", params={"limit": 200}))
    needle = name.lower()
    for m in data["items"]:
        if m["name"].lower() == needle:
            return m["id"]
    console.print(f"[red]Model not found: {name}[/red]")
    raise typer.Exit(1)


def resolve_secret(client: httpx.Client, name: str) -> str:
    """Resolve a secret name to a secret ID."""
    data = handle_response(client.get("/admin/secrets", params={"limit": 200}))
    needle = name.lower()
    for s in data["items"]:
        if s["name"].lower() == needle:
            return s["id"]
    console.print(f"[red]Secret not found: {name}[/red]")
    raise typer.Exit(1)


def resolve_rule(client: httpx.Client, name: str) -> str:
    """Resolve a rule name to a rule ID."""
    data = handle_response(client.get("/admin/rules", params={"limit": 200}))
    needle = name.lower()
    for r in data["items"]:
        if r["name"].lower() == needle:
            return r["id"]
    console.print(f"[red]Rule not found: {name}[/red]")
    raise typer.Exit(1)
