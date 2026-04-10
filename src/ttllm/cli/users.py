"""CLI commands for user management."""

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
    resolve_user,
)

app = typer.Typer(help="Manage users")


@app.command("list")
def users_list(
    offset: int = typer.Option(0, help="Offset for pagination"),
    limit: int = typer.Option(50, help="Limit for pagination"),
    as_json: bool = JSON_OPTION,
):
    """List all users."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/users", params={"offset": offset, "limit": limit})
        )

    if as_json:
        print_json(data)
        return

    table = Table(title="Users")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Email")
    table.add_column("Groups")
    table.add_column("IdP")
    table.add_column("Active")
    table.add_column("Created")

    for u in data["items"]:
        table.add_row(
            u["id"][:8] + "...",
            u["name"],
            u["email"],
            ", ".join(u.get("groups", [])) or "-",
            u.get("identity_provider") or "internal",
            "Yes" if u["is_active"] else "No",
            u["created_at"][:10],
        )
    console.print(table)
    console.print(f"Total: {data['total']}")


@app.command("create")
def users_create(
    name: str = typer.Option(..., help="User name"),
    email: str = typer.Option(..., help="User email"),
    password: Optional[str] = typer.Option(None, help="Password (for internal users)"),
):
    """Create a new user."""
    body = {"name": name, "email": email}
    if password:
        body["password"] = password
    with get_client() as client:
        data = handle_response(client.post("/admin/users", json=body))
    console.print(f"[green]User created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Email: {data['email']}")


@app.command("show")
def users_show(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show user details including groups and permissions."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        data = handle_response(client.get(f"/admin/users/{user_id}"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]User:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Email: {data['email']}")
    console.print(f"  IdP: {data.get('identity_provider') or 'internal'}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:10]}")

    groups = data.get("groups", [])
    if groups:
        console.print(f"\n[bold]Groups:[/bold] {', '.join(groups)}")
    else:
        console.print("\n[bold]Groups:[/bold] (none)")


@app.command("models")
def users_models(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """List all models a user can access (direct + group assignments)."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        data = handle_response(client.get(f"/admin/users/{user_id}/models"))

    if as_json:
        print_json(data)
        return

    if not data:
        console.print("[yellow]No models assigned to this user.[/yellow]")
        return

    table = Table(title="Accessible Models")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Provider Model ID")
    table.add_column("Active")

    for m in data:
        table.add_row(
            m["id"][:8] + "...",
            m["name"],
            m["provider"],
            m["provider_model_id"],
            "Yes" if m["is_active"] else "No",
        )
    console.print(table)


@app.command("update")
def users_update(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New name"),
    email: Optional[str] = typer.Option(None, "--email", help="New email"),
    password: Optional[str] = typer.Option(None, "--password", help="New password"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Update a user's details."""
    body: dict = {}
    if name is not None:
        body["name"] = name
    if email is not None:
        body["email"] = email
    if password is not None:
        body["password"] = password
    if not body:
        console.print("[red]Nothing to update. Provide --name, --email, or --password.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        data = handle_response(client.patch(f"/admin/users/{user_id}", json=body))
    console.print(f"[green]User updated:[/green] {data['name']}")


@app.command("delete")
def users_delete(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Deactivate a user."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        resp = client.delete(f"/admin/users/{user_id}")
        if resp.status_code == 204:
            console.print("[green]User deactivated.[/green]")
        else:
            handle_response(resp)


@app.command("permissions")
def users_permissions(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show a user's direct and effective permissions."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        data = handle_response(client.get(f"/admin/users/{user_id}/permissions"))

    if as_json:
        print_json(data)
        return

    console.print("[bold]Direct permissions:[/bold]")
    for p in data.get("direct_permissions", []):
        console.print(f"  - {p}")
    if not data.get("direct_permissions"):
        console.print("  (none)")

    console.print("\n[bold]Effective permissions (direct + groups):[/bold]")
    for p in data.get("effective_permissions", []):
        console.print(f"  - {p}")


@app.command("add-permission")
def users_add_permission(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    permission: list[str] = typer.Option(..., "--permission", help="Permission(s) to assign"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Assign permission(s) directly to a user."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        data = handle_response(
            client.post(f"/admin/users/{user_id}/permissions", json={"permissions": permission})
        )
    for r in data.get("permissions", []):
        console.print(f"  {r['permission']}: {r['status']}")


@app.command("remove-permission")
def users_remove_permission(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    permission: str = typer.Option(..., "--permission", help="Permission to remove"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Remove a direct permission from a user."""
    with get_client() as client:
        user_id = user if use_ids else resolve_user(client, user)
        resp = client.delete(f"/admin/users/{user_id}/permissions/{permission}")
        if resp.status_code == 204:
            console.print(f"[green]Permission '{permission}' removed from user.[/green]")
        else:
            handle_response(resp)