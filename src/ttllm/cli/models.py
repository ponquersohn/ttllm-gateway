"""CLI commands for model management."""

from __future__ import annotations

import json
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
    resolve_model,
    resolve_user,
)

app = typer.Typer(help="Manage models")


@app.command("list")
def models_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    as_json: bool = JSON_OPTION,
):
    """List all models."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/models", params={"offset": offset, "limit": limit})
        )

    if as_json:
        print_json(data)
        return

    table = Table(title="Models")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Provider Model ID")
    table.add_column("Input $/1K")
    table.add_column("Output $/1K")
    table.add_column("Active")

    for m in data["items"]:
        table.add_row(
            m["id"][:8] + "...",
            m["name"],
            m["provider"],
            m["provider_model_id"],
            str(m["input_cost_per_1k"]),
            str(m["output_cost_per_1k"]),
            "Yes" if m["is_active"] else "No",
        )
    console.print(table)


@app.command("show")
def models_show(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show model details."""
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        data = handle_response(client.get(f"/admin/models/{model_id}"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]Model:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Provider: {data['provider']}")
    console.print(f"  Provider Model ID: {data['provider_model_id']}")
    console.print(f"  Input cost/1K: {data['input_cost_per_1k']}")
    console.print(f"  Output cost/1K: {data['output_cost_per_1k']}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:10]}")
    if data.get("config_json"):
        console.print(f"  Config: {json.dumps(data['config_json'])}")


@app.command("create")
def models_create(
    name: str = typer.Option(..., help="Model display name"),
    provider: str = typer.Option(..., help="Provider (bedrock, openai, etc.)"),
    provider_model_id: str = typer.Option(..., help="Provider-specific model ID"),
    input_cost: float = typer.Option(0.0, help="Cost per 1K input tokens"),
    output_cost: float = typer.Option(0.0, help="Cost per 1K output tokens"),
    config: Optional[str] = typer.Option(None, help="JSON config string"),
):
    """Add a new model."""
    config_json = json.loads(config) if config else {}
    with get_client() as client:
        data = handle_response(
            client.post(
                "/admin/models",
                json={
                    "name": name,
                    "provider": provider,
                    "provider_model_id": provider_model_id,
                    "config_json": config_json,
                    "input_cost_per_1k": str(input_cost),
                    "output_cost_per_1k": str(output_cost),
                },
            )
        )
    console.print(f"[green]Model created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Provider: {data['provider']}")


@app.command("update")
def models_update(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New display name"),
    provider: Optional[str] = typer.Option(None, "--provider", help="New provider"),
    provider_model_id: Optional[str] = typer.Option(None, "--provider-model-id", help="New provider model ID"),
    config: Optional[str] = typer.Option(None, "--config", help="New JSON config (replaces existing)"),
    input_cost: Optional[float] = typer.Option(None, "--input-cost", help="Cost per 1K input tokens"),
    output_cost: Optional[float] = typer.Option(None, "--output-cost", help="Cost per 1K output tokens"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat model argument as UUID"),
):
    """Update an existing model."""
    body: dict = {}
    if name is not None:
        body["name"] = name
    if provider is not None:
        body["provider"] = provider
    if provider_model_id is not None:
        body["provider_model_id"] = provider_model_id
    if config is not None:
        body["config_json"] = json.loads(config)
    if input_cost is not None:
        body["input_cost_per_1k"] = str(input_cost)
    if output_cost is not None:
        body["output_cost_per_1k"] = str(output_cost)
    if not body:
        console.print("[red]Nothing to update. Provide at least one option.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        data = handle_response(client.patch(f"/admin/models/{model_id}", json=body))
    console.print(f"[green]Model updated:[/green] {data['name']}")


@app.command("delete")
def models_delete(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat model argument as UUID"),
):
    """Deactivate a model."""
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        resp = client.delete(f"/admin/models/{model_id}")
        if resp.status_code == 204:
            console.print("[green]Model deactivated.[/green]")
        else:
            handle_response(resp)


@app.command("assign")
def models_assign(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    user: Optional[list[str]] = typer.Option(None, "--user", help="User name(s) to assign"),
    group: Optional[list[str]] = typer.Option(None, "--group", help="Group name(s) to assign"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat arguments as UUIDs instead of names"),
):
    """Assign a model to user(s) or group(s)."""
    if not user and not group:
        console.print("[red]Provide --user or --group.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        if user:
            user_ids = user if use_ids else [resolve_user(client, u) for u in user]
            data = handle_response(
                client.post(
                    f"/admin/models/{model_id}/assign",
                    json={"user_ids": user_ids},
                )
            )
            for a in data.get("assignments", []):
                console.print(f"  User {a['user_id'][:8]}...: {a['status']}")
        if group:
            group_ids = group if use_ids else [resolve_group(client, g) for g in group]
            data = handle_response(
                client.post(
                    f"/admin/models/{model_id}/assign-group",
                    json={"group_ids": group_ids},
                )
            )
            for a in data.get("assignments", []):
                console.print(f"  Group {a['group_id'][:8]}...: {a['status']}")


@app.command("unassign")
def models_unassign(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    user: Optional[str] = typer.Option(None, "--user", help="User name to unassign"),
    group: Optional[str] = typer.Option(None, "--group", help="Group name to unassign"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat arguments as UUIDs instead of names"),
):
    """Remove model assignment from a user or group."""
    if not user and not group:
        console.print("[red]Provide --user or --group.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        if user:
            user_id = user if use_ids else resolve_user(client, user)
            resp = client.delete(f"/admin/models/{model_id}/assign/{user_id}")
            if resp.status_code == 204:
                console.print("[green]User assignment removed.[/green]")
            else:
                handle_response(resp)
        if group:
            group_id = group if use_ids else resolve_group(client, group)
            resp = client.delete(f"/admin/models/{model_id}/assign-group/{group_id}")
            if resp.status_code == 204:
                console.print("[green]Group assignment removed.[/green]")
            else:
                handle_response(resp)
