"""CLI commands for secret management."""

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
    resolve_secret,
)

app = typer.Typer(help="Manage secrets")


@app.command("list")
def secrets_list(
    offset: int = typer.Option(0, help="Offset for pagination"),
    limit: int = typer.Option(50, help="Limit for pagination"),
    as_json: bool = JSON_OPTION,
):
    """List all secrets (values are never shown)."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/secrets", params={"offset": offset, "limit": limit})
        )

    if as_json:
        print_json(data)
        return

    table = Table(title="Secrets")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Created")
    table.add_column("Updated")

    for s in data["items"]:
        table.add_row(
            s["id"][:8] + "...",
            s["name"],
            s.get("description") or "-",
            s["created_at"][:10],
            s["updated_at"][:10],
        )
    console.print(table)
    console.print(f"Total: {data['total']}")


@app.command("create")
def secrets_create(
    name: str = typer.Option(..., "--name", help="Secret name"),
    value: Optional[str] = typer.Option(None, "--value", help="Secret value (if omitted, prompted with hidden input)"),
    description: Optional[str] = typer.Option(None, "--description", help="Description"),
):
    """Create a new secret. Value is prompted with hidden input."""
    if value is None:
        value = typer.prompt("Secret value", hide_input=True)
    body: dict = {"name": name, "value": value}
    if description:
        body["description"] = description
    with get_client() as client:
        data = handle_response(client.post("/admin/secrets", json=body))
    console.print(f"[green]Secret created:[/green] {data['name']}")


@app.command("show")
def secrets_show(
    name: str = typer.Argument(help="Secret name"),
    as_json: bool = JSON_OPTION,
):
    """Show secret metadata (value is never displayed)."""
    with get_client() as client:
        secret_id = resolve_secret(client, name)
        data = handle_response(client.get(f"/admin/secrets/{secret_id}"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]Secret:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  Created: {data['created_at'][:19]}")
    console.print(f"  Updated: {data['updated_at'][:19]}")


@app.command("update")
def secrets_update(
    name: str = typer.Argument(help="Secret name"),
    prompt_value: bool = typer.Option(False, "--prompt-value", help="Prompt for a new secret value"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
):
    """Update a secret's value or description."""
    body: dict = {}
    if prompt_value:
        body["value"] = typer.prompt("New secret value", hide_input=True)
    if description is not None:
        body["description"] = description
    if not body:
        console.print("[red]Nothing to update. Provide --prompt-value or --description.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        secret_id = resolve_secret(client, name)
        data = handle_response(client.patch(f"/admin/secrets/{secret_id}", json=body))
    console.print(f"[green]Secret updated:[/green] {data['name']}")


@app.command("delete")
def secrets_delete(
    name: str = typer.Argument(help="Secret name"),
):
    """Delete a secret."""
    with get_client() as client:
        secret_id = resolve_secret(client, name)
        resp = client.delete(f"/admin/secrets/{secret_id}")
        if resp.status_code == 204:
            console.print("[green]Secret deleted.[/green]")
        else:
            handle_response(resp)
