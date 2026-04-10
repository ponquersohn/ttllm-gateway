"""CLI commands for self-service (current user's models and tokens)."""

from __future__ import annotations

from typing import Optional

import typer
from rich.table import Table

from ttllm.cli._common import (
    JSON_OPTION,
    console,
    get_client,
    handle_response,
    print_json,
)

app = typer.Typer(help="Self-service: view your models and manage your tokens")
tokens_app = typer.Typer(help="Manage your own tokens")
app.add_typer(tokens_app, name="tokens")


@app.command("models")
def me_models(
    as_json: bool = JSON_OPTION,
):
    """List models available to you (direct + group assignments)."""
    with get_client() as client:
        data = handle_response(client.get("/me/models"))

    if as_json:
        print_json(data)
        return

    if not data:
        console.print("[yellow]No models available.[/yellow]")
        return

    table = Table(title="My Models")
    table.add_column("Name")
    table.add_column("Provider")

    for m in data:
        table.add_row(m["name"], m["provider"])
    console.print(table)


@tokens_app.command("list")
@tokens_app.callback(invoke_without_command=True)
def me_tokens_list(
    as_json: bool = JSON_OPTION,
):
    """List your active tokens."""
    with get_client() as client:
        data = handle_response(client.get("/me/tokens"))

    if as_json:
        print_json(data)
        return

    if not data:
        console.print("[yellow]No tokens found.[/yellow]")
        return

    table = Table(title="My Tokens")
    table.add_column("ID", style="dim")
    table.add_column("Label")
    table.add_column("Permissions")
    table.add_column("Active")
    table.add_column("Expires")

    for t in data:
        table.add_row(
            t["id"][:8] + "...",
            t.get("label") or "-",
            ", ".join(t.get("permissions", [])) or "-",
            "Yes" if t["is_active"] else "No",
            (t.get("expires_at") or "never")[:10],
        )
    console.print(table)


@tokens_app.command("create")
def me_tokens_create(
    label: Optional[str] = typer.Option(None, "--label", help="Token label"),
    ttl_days: Optional[int] = typer.Option(None, "--ttl-days", help="Token lifetime in days"),
    permissions: Optional[str] = typer.Option(None, "--permissions", help="Comma-separated permissions (default: llm.invoke)"),
):
    """Create a token for yourself."""
    body: dict = {}
    if label:
        body["label"] = label
    if ttl_days is not None:
        body["ttl_days"] = ttl_days
    if permissions:
        body["permissions"] = [s.strip() for s in permissions.split(",") if s.strip()]
    with get_client() as client:
        data = handle_response(client.post("/me/tokens", json=body))
    console.print("[green]Token created:[/green]")
    console.print(f"  Token: [bold]{data['access_token']}[/bold]")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Permissions: {', '.join(data.get('permissions', []))}")
    console.print(f"  Label: {data.get('label') or 'N/A'}")
    console.print(f"  Expires: {data.get('expires_at') or 'never'}")
    console.print("[yellow]Save this token now -- it will not be shown again.[/yellow]")


@tokens_app.command("delete")
def me_tokens_delete(
    token_id: str = typer.Argument(help="Token ID to revoke"),
):
    """Revoke one of your own tokens."""
    with get_client() as client:
        resp = client.delete(f"/me/tokens/{token_id}")
        if resp.status_code == 204:
            console.print("[green]Token revoked.[/green]")
        else:
            handle_response(resp)
