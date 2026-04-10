"""CLI commands for group management."""

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
    resolve_group,
)

app = typer.Typer(help="Manage groups")


@app.command("list")
def groups_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    as_json: bool = JSON_OPTION,
):
    """List all groups."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/groups", params={"offset": offset, "limit": limit})
        )

    if as_json:
        print_json(data)
        return

    table = Table(title="Groups")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Permissions")
    table.add_column("Active")

    for g in data["items"]:
        table.add_row(
            g["id"][:8] + "...",
            g["name"],
            g.get("description") or "-",
            ", ".join(g.get("permissions", [])) or "-",
            "Yes" if g["is_active"] else "No",
        )
    console.print(table)


@app.command("create")
def groups_create(
    name: str = typer.Option(..., help="Group name"),
    description: Optional[str] = typer.Option(None, help="Group description"),
):
    """Create a new group."""
    body = {"name": name}
    if description:
        body["description"] = description
    with get_client() as client:
        data = handle_response(client.post("/admin/groups", json=body))
    console.print(f"[green]Group created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")


@app.command("show")
def groups_show(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show group details."""
    with get_client() as client:
        group_id = group if use_ids else resolve_group(client, group)
        data = handle_response(client.get(f"/admin/groups/{group_id}"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]Group:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:10]}")
    perms = data.get("permissions", [])
    if perms:
        console.print(f"\n[bold]Permissions:[/bold] {', '.join(perms)}")
    else:
        console.print("\n[bold]Permissions:[/bold] (none)")


@app.command("update")
def groups_update(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New group name"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Update a group's name or description."""
    body: dict = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if not body:
        console.print("[red]Nothing to update. Provide --name or --description.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        group_id = group if use_ids else resolve_group(client, group)
        data = handle_response(client.patch(f"/admin/groups/{group_id}", json=body))
    console.print(f"[green]Group updated:[/green] {data['name']}")


@app.command("delete")
def groups_delete(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Delete a group."""
    with get_client() as client:
        group_id = group if use_ids else resolve_group(client, group)
        resp = client.delete(f"/admin/groups/{group_id}")
        if resp.status_code == 204:
            console.print("[green]Group deleted.[/green]")
        else:
            handle_response(resp)


@app.command("add-permission")
def groups_add_permission(
    group_id: str = typer.Argument(help="Group ID"),
    permission: str = typer.Option(..., "--permission", help="Permission to assign"),
):
    """Assign a permission to a group."""
    with get_client() as client:
        handle_response(
            client.post(f"/admin/groups/{group_id}/permissions", json={"permission": permission})
        )
    console.print(f"[green]Permission '{permission}' assigned to group.[/green]")


@app.command("remove-permission")
def groups_remove_permission(
    group_id: str = typer.Argument(help="Group ID"),
    permission: str = typer.Option(..., "--permission", help="Permission to remove"),
):
    """Remove a permission from a group."""
    with get_client() as client:
        resp = client.delete(f"/admin/groups/{group_id}/permissions/{permission}")
        if resp.status_code == 204:
            console.print(f"[green]Permission '{permission}' removed from group.[/green]")
        else:
            handle_response(resp)


@app.command("add-member")
def groups_add_member(
    group_id: str = typer.Argument(help="Group ID"),
    user: list[str] = typer.Option(..., "--user", help="User ID(s) to add"),
):
    """Add user(s) to a group."""
    with get_client() as client:
        data = handle_response(
            client.post(f"/admin/groups/{group_id}/members", json={"user_ids": user})
        )
    for m in data.get("members", []):
        console.print(f"  User {m['user_id'][:8]}...: {m['status']}")


@app.command("remove-member")
def groups_remove_member(
    group_id: str = typer.Argument(help="Group ID"),
    user: str = typer.Option(..., "--user", help="User ID to remove"),
):
    """Remove a user from a group."""
    with get_client() as client:
        resp = client.delete(f"/admin/groups/{group_id}/members/{user}")
        if resp.status_code == 204:
            console.print("[green]Member removed.[/green]")
        else:
            handle_response(resp)
