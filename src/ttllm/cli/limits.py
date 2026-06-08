"""CLI commands for token quota limit management."""

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

app = typer.Typer(help="Manage token quota limits")


@app.command("list")
def limits_list(
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Filter by user UUID"),
    group_id: Optional[str] = typer.Option(None, "--group-id", help="Filter by group UUID"),
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    as_json: bool = JSON_OPTION,
):
    """List token quota limits."""
    params: dict = {"offset": offset, "limit": limit}
    if user_id:
        params["user_id"] = user_id
    if group_id:
        params["group_id"] = group_id
    with get_client() as client:
        data = handle_response(client.get("/admin/usage-limits", params=params))

    if as_json:
        print_json(data)
        return

    table = Table(title="Quota Limits")
    table.add_column("ID", style="dim")
    table.add_column("Scope")
    table.add_column("Window")
    table.add_column("Token Cap", justify="right")
    table.add_column("Window Secs", justify="right")
    table.add_column("Target", style="dim")

    for item in data["items"]:
        target = item.get("user_id") or item.get("group_id") or "-"
        if target != "-":
            target = target[:8] + "..."
        ws = item.get("window_seconds")
        table.add_row(
            item["id"][:8] + "...",
            item["scope"],
            item["window_kind"],
            f"{item['token_cap']:,}",
            str(ws) if ws is not None else "(default)",
            target,
        )
    console.print(table)
    console.print(f"Total: {data['total']}")


@app.command("show")
def limits_show(
    limit_id: str = typer.Argument(help="Limit UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show a single quota limit."""
    with get_client() as client:
        data = handle_response(client.get(f"/admin/usage-limits/{limit_id}"))

    if as_json:
        print_json(data)
        return

    ws = data.get("window_seconds")
    console.print(f"[bold]Limit:[/bold] {data['id']}")
    console.print(f"  Scope:        {data['scope']}")
    console.print(f"  Window kind:  {data['window_kind']}")
    console.print(f"  Token cap:    {data['token_cap']:,}")
    console.print(f"  Window secs:  {ws if ws is not None else '(default for window kind)'}")
    console.print(f"  User ID:      {data.get('user_id') or '-'}")
    console.print(f"  Group ID:     {data.get('group_id') or '-'}")
    console.print(f"  Created:      {data['created_at'][:19]}")


@app.command("create")
def limits_create(
    scope: str = typer.Option(..., "--scope", help="Scope: user, group, or global"),
    window_kind: str = typer.Option(..., "--window-kind", help="Window kind: 5h, weekly, or monthly"),
    token_cap: int = typer.Option(..., "--token-cap", help="Maximum tokens allowed in the window"),
    window_seconds: Optional[int] = typer.Option(
        None, "--window-seconds", help="Window length in seconds (omit to use the default for the window kind)"
    ),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User UUID (required when scope=user)"),
    group_id: Optional[str] = typer.Option(None, "--group-id", help="Group UUID (required when scope=group)"),
):
    """Create a quota limit."""
    body: dict = {"scope": scope, "window_kind": window_kind, "token_cap": token_cap}
    if window_seconds is not None:
        body["window_seconds"] = window_seconds
    if user_id:
        body["user_id"] = user_id
    if group_id:
        body["group_id"] = group_id
    with get_client() as client:
        data = handle_response(client.post("/admin/usage-limits", json=body))
    console.print(f"[green]Limit created:[/green] {data['id']}")
    console.print(f"  {data['scope']} / {data['window_kind']} / cap {data['token_cap']:,}")


@app.command("update")
def limits_update(
    limit_id: str = typer.Argument(help="Limit UUID"),
    token_cap: Optional[int] = typer.Option(None, "--token-cap", help="New token cap"),
    window_seconds: Optional[int] = typer.Option(None, "--window-seconds", help="New window length in seconds"),
):
    """Update a quota limit's cap and/or window length."""
    body: dict = {}
    if token_cap is not None:
        body["token_cap"] = token_cap
    if window_seconds is not None:
        body["window_seconds"] = window_seconds
    if not body:
        console.print("[red]Nothing to update. Provide --token-cap and/or --window-seconds.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        data = handle_response(client.patch(f"/admin/usage-limits/{limit_id}", json=body))
    console.print(f"[green]Limit updated:[/green] cap {data['token_cap']:,}")


@app.command("delete")
def limits_delete(
    limit_id: str = typer.Argument(help="Limit UUID"),
):
    """Delete a quota limit."""
    with get_client() as client:
        resp = client.delete(f"/admin/usage-limits/{limit_id}")
        if resp.status_code == 204:
            console.print("[green]Limit deleted.[/green]")
        else:
            handle_response(resp)
