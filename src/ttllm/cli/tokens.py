"""CLI commands for token management."""

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

app = typer.Typer(help="Manage tokens")


@app.command("create")
def tokens_create(
    user_id: Optional[str] = typer.Option(None, "--user", help="User ID (defaults to current user)"),
    label: Optional[str] = typer.Option(None, "--label", help="Token label"),
    ttl_days: Optional[int] = typer.Option(None, "--ttl-days", help="Token lifetime in days (default: 30, max: 365)"),
    permissions: Optional[str] = typer.Option(None, "--permissions", help="Comma-separated permissions (default: llm.invoke)"),
):
    """Generate a token with specified permissions (default: llm.invoke for gateway access)."""
    body: dict = {}
    if user_id:
        body["user_id"] = user_id
    if label:
        body["label"] = label
    if ttl_days is not None:
        body["ttl_days"] = ttl_days
    if permissions:
        body["permissions"] = [s.strip() for s in permissions.split(",") if s.strip()]
    with get_client() as client:
        data = handle_response(client.post("/admin/tokens", json=body))
    console.print("[green]Token created:[/green]")
    console.print(f"  Token: [bold]{data['access_token']}[/bold]")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Permissions: {', '.join(data.get('permissions', []))}")
    console.print(f"  Label: {data.get('label') or 'N/A'}")
    console.print(f"  Expires: {data.get('expires_at') or 'never'}")
    console.print("[yellow]Save this token now -- it will not be shown again.[/yellow]")


@app.command("show")
def tokens_show(
    token_id: str = typer.Argument(help="Token ID"),
    as_json: bool = JSON_OPTION,
):
    """Show token details (token value is never displayed)."""
    with get_client() as client:
        data = handle_response(client.get(f"/admin/tokens/{token_id}"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]Token:[/bold] {data['id'][:5]}...")
    console.print(f"  ID: {data['id']}")
    console.print(f"  User: {data.get('user_email') or data['user_id']}")
    console.print(f"  Label: {data.get('label') or '(none)'}")
    console.print(f"  Permissions: {', '.join(data.get('permissions', []))}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:19]}")
    console.print(f"  Expires: {(data.get('expires_at') or 'never')[:19]}")


@app.command("list")
def tokens_list(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    as_json: bool = JSON_OPTION,
):
    """List active tokens."""
    params = {}
    if user_id:
        params["user_id"] = user_id
    with get_client() as client:
        data = handle_response(client.get("/admin/tokens", params=params))

    if as_json:
        print_json(data)
        return

    table = Table(title="Tokens")
    table.add_column("ID", style="dim")
    table.add_column("User ID", style="dim")
    table.add_column("Email")
    table.add_column("Label")
    table.add_column("Permissions")
    table.add_column("Active")
    table.add_column("Created")
    table.add_column("Expires")

    for t in data:
        table.add_row(
            t["id"][:8] + "...",
            t["user_id"][:8] + "...",
            t.get("user_email") or "-",
            t.get("label") or "-",
            ", ".join(t.get("permissions", [])) or "-",
            "Yes" if t["is_active"] else "No",
            t["created_at"][:10],
            (t.get("expires_at") or "never")[:10],
        )
    console.print(table)


@app.command("delete")
def tokens_delete(
    token_id: str = typer.Argument(help="Token ID to revoke"),
):
    """Revoke a token."""
    with get_client() as client:
        resp = client.delete(f"/admin/tokens/{token_id}")
        if resp.status_code == 204:
            console.print("[green]Token revoked.[/green]")
        else:
            handle_response(resp)
