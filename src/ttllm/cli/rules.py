"""CLI commands for rule management."""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.table import Table

from ttllm.cli._common import (
    TtllmTyper,
    console,
    get_client,
    handle_response,
    json_mode,
    print_json,
    resolve_rule,
)

app = TtllmTyper(help="Manage rules")


@app.command("list")
def rules_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
):
    """List all rules."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/rules", params={"offset": offset, "limit": limit})
        )

    if json_mode():
        print_json(data)
        return

    table = Table(title="Rules")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Weight")
    table.add_column("Enabled")
    table.add_column("Action")
    table.add_column("Description")

    for r in data["items"]:
        table.add_row(
            r["id"][:8] + "...",
            r["name"],
            str(r["weight"]),
            "Yes" if r["enabled"] else "No",
            r["action"].get("type", ""),
            (r.get("description") or "")[:40],
        )
    console.print(table)


@app.command("show")
def rules_show(
    rule: str = typer.Argument(help="Rule name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Show rule details."""
    with get_client() as client:
        rule_id = rule if use_ids else resolve_rule(client, rule)
        data = handle_response(client.get(f"/admin/rules/{rule_id}"))

    if json_mode():
        print_json(data)
        return

    console.print(f"[bold]Rule:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Weight: {data['weight']}")
    console.print(f"  Enabled: {'Yes' if data['enabled'] else 'No'}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  Created: {data['created_at'][:10]}")
    console.print(f"  Conditions: {json.dumps(data['conditions'], indent=2)}")
    console.print(f"  Action: {json.dumps(data['action'], indent=2)}")


@app.command("create")
def rules_create(
    name: str = typer.Option(..., help="Rule name"),
    conditions: str = typer.Option(..., help="JSON conditions object"),
    action: str = typer.Option(..., help="JSON action object"),
    weight: int = typer.Option(0, help="Rule weight (higher = evaluated first)"),
    description: Optional[str] = typer.Option(None, help="Rule description"),
    disabled: bool = typer.Option(False, "--disabled", help="Create rule as disabled"),
):
    """Create a new rule."""
    body: dict = {
        "name": name,
        "conditions": json.loads(conditions),
        "action": json.loads(action),
        "weight": weight,
        "enabled": not disabled,
    }
    if description is not None:
        body["description"] = description
    with get_client() as client:
        data = handle_response(client.post("/admin/rules", json=body))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Rule created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Weight: {data['weight']}")


@app.command("update")
def rules_update(
    rule: str = typer.Argument(help="Rule name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New name"),
    weight: Optional[int] = typer.Option(None, "--weight", help="New weight"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    conditions: Optional[str] = typer.Option(None, "--conditions", help="New JSON conditions"),
    action: Optional[str] = typer.Option(None, "--action", help="New JSON action"),
    enabled: Optional[bool] = typer.Option(None, "--enabled", help="Enable/disable rule"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat rule argument as UUID"),
):
    """Update an existing rule."""
    body: dict = {}
    if name is not None:
        body["name"] = name
    if weight is not None:
        body["weight"] = weight
    if description is not None:
        body["description"] = description
    if conditions is not None:
        body["conditions"] = json.loads(conditions)
    if action is not None:
        body["action"] = json.loads(action)
    if enabled is not None:
        body["enabled"] = enabled
    if not body:
        console.print("[red]Nothing to update. Provide at least one option.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        rule_id = rule if use_ids else resolve_rule(client, rule)
        data = handle_response(client.patch(f"/admin/rules/{rule_id}", json=body))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Rule updated:[/green] {data['name']}")


@app.command("delete")
def rules_delete(
    rule: str = typer.Argument(help="Rule name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat rule argument as UUID"),
):
    """Delete a rule."""
    with get_client() as client:
        rule_id = rule if use_ids else resolve_rule(client, rule)
        resp = client.delete(f"/admin/rules/{rule_id}")
        if resp.status_code == 204:
            if json_mode():
                print_json({"status": "deleted", "id": rule_id})
            else:
                console.print("[green]Rule deleted.[/green]")
        else:
            handle_response(resp)
