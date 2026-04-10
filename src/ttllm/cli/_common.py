"""Shared CLI utilities: console, client helpers, name resolution."""

from __future__ import annotations

import json

import httpx
import typer
from rich.console import Console

from ttllm.cli.client import TTLLMClient

console = Console()

JSON_OPTION = typer.Option(False, "--json", help="Output raw JSON")


def print_json(data) -> None:
    """Print data as formatted JSON and exit."""
    console.print(json.dumps(data, indent=2, default=str), highlight=False)


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