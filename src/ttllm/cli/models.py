"""CLI commands for model management."""

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
    resolve_group,
    resolve_model,
    resolve_user,
)

app = TtllmTyper(help="Manage models")


@app.command("list")
def models_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
):
    """List all models."""
    with get_client() as client:
        data = handle_response(
            client.get("/admin/models", params={"offset": offset, "limit": limit})
        )

    if json_mode():
        print_json(data)
        return

    table = Table(title="Models")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Provider")
    table.add_column("Provider Model ID")
    table.add_column("Match Pattern")
    table.add_column("Input $/1K")
    table.add_column("Output $/1K")
    table.add_column("Cache R $/1K")
    table.add_column("Cache W $/1K")
    table.add_column("Active")

    for m in data["items"]:
        table.add_row(
            m["id"][:8] + "...",
            m["name"],
            m["provider"],
            m["provider_model_id"],
            m.get("match_pattern") or "",
            str(m["input_cost_per_1k"]),
            str(m["output_cost_per_1k"]),
            str(m.get("cache_read_cost_per_1k", "0")),
            str(m.get("cache_write_cost_per_1k", "0")),
            "Yes" if m["is_active"] else "No",
        )
    console.print(table)


@app.command("show")
def models_show(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Show model details."""
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        data = handle_response(client.get(f"/admin/models/{model_id}"))

    if json_mode():
        print_json(data)
        return

    console.print(f"[bold]Model:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Provider: {data['provider']}")
    console.print(f"  Provider Model ID: {data['provider_model_id']}")
    console.print(f"  Input cost/1K: {data['input_cost_per_1k']}")
    console.print(f"  Output cost/1K: {data['output_cost_per_1k']}")
    console.print(f"  Cache read cost/1K: {data.get('cache_read_cost_per_1k', '0')}")
    console.print(f"  Cache write cost/1K: {data.get('cache_write_cost_per_1k', '0')}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:10]}")
    if data.get("match_pattern"):
        console.print(f"  Match Pattern: {data['match_pattern']}")
    if data.get("config_json"):
        console.print(f"  Config: {json.dumps(data['config_json'])}")


@app.command("create")
def models_create(
    name: str = typer.Option(..., help="Model display name"),
    provider: str = typer.Option(..., help="Provider (bedrock, openai, etc.)"),
    provider_model_id: str = typer.Option(..., help="Provider-specific model ID"),
    input_cost: float = typer.Option(0.0, help="Cost per 1K input tokens"),
    output_cost: float = typer.Option(0.0, help="Cost per 1K output tokens"),
    cache_read_cost: float = typer.Option(0.0, "--cache-read-cost", help="Cost per 1K cache-read input tokens"),
    cache_write_cost: float = typer.Option(0.0, "--cache-write-cost", help="Cost per 1K cache-write input tokens"),
    config: Optional[str] = typer.Option(None, help="JSON config string"),
    match_pattern: Optional[str] = typer.Option(None, "--match-pattern", help="Regex pattern for flexible model name matching"),
):
    """Add a new model."""
    config_json = json.loads(config) if config else {}
    body: dict = {
        "name": name,
        "provider": provider,
        "provider_model_id": provider_model_id,
        "config_json": config_json,
        "input_cost_per_1k": str(input_cost),
        "output_cost_per_1k": str(output_cost),
        "cache_read_cost_per_1k": str(cache_read_cost),
        "cache_write_cost_per_1k": str(cache_write_cost),
    }
    if match_pattern is not None:
        body["match_pattern"] = match_pattern
    with get_client() as client:
        data = handle_response(client.post("/admin/models", json=body))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Model created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Provider: {data['provider']}")


@app.command("update")
def models_update(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New display name"),
    provider: Optional[str] = typer.Option(None, "--provider", help="New provider"),
    provider_model_id: Optional[str] = typer.Option(None, "--provider-model-id", help="New provider model ID"),
    config: Optional[str] = typer.Option(None, "--config", help="JSON config (replaces existing; use --merge-config to shallow-merge)"),
    merge_config: bool = typer.Option(False, "--merge-config", help="Merge into existing config instead of replacing"),
    input_cost: Optional[float] = typer.Option(None, "--input-cost", help="Cost per 1K input tokens"),
    output_cost: Optional[float] = typer.Option(None, "--output-cost", help="Cost per 1K output tokens"),
    cache_read_cost: Optional[float] = typer.Option(None, "--cache-read-cost", help="Cost per 1K cache-read input tokens"),
    cache_write_cost: Optional[float] = typer.Option(None, "--cache-write-cost", help="Cost per 1K cache-write input tokens"),
    match_pattern: Optional[str] = typer.Option(None, "--match-pattern", help="Regex pattern (use empty string to clear)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat model argument as UUID"),
):
    """Update an existing model."""
    if merge_config and config is None:
        console.print("[red]--merge-config requires --config[/red]")
        raise typer.Exit(1)
    body: dict = {}
    if name is not None:
        body["name"] = name
    if provider is not None:
        body["provider"] = provider
    if provider_model_id is not None:
        body["provider_model_id"] = provider_model_id
    if config is not None:
        body["config_json"] = json.loads(config)
        if merge_config:
            body["merge_config"] = True
    if input_cost is not None:
        body["input_cost_per_1k"] = str(input_cost)
    if output_cost is not None:
        body["output_cost_per_1k"] = str(output_cost)
    if cache_read_cost is not None:
        body["cache_read_cost_per_1k"] = str(cache_read_cost)
    if cache_write_cost is not None:
        body["cache_write_cost_per_1k"] = str(cache_write_cost)
    if match_pattern is not None:
        body["match_pattern"] = match_pattern if match_pattern != "" else None
    if not body:
        console.print("[red]Nothing to update. Provide at least one option.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        data = handle_response(client.patch(f"/admin/models/{model_id}", json=body))
    if json_mode():
        print_json(data)
        return
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
            if json_mode():
                print_json({"status": "deactivated", "id": model_id})
            else:
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
    result: dict = {}
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
            result["users"] = data.get("assignments", [])
            if not json_mode():
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
            result["groups"] = data.get("assignments", [])
            if not json_mode():
                for a in data.get("assignments", []):
                    console.print(f"  Group {a['group_id'][:8]}...: {a['status']}")
    if json_mode():
        print_json(result)


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
    result: dict = {}
    with get_client() as client:
        model_id = model if use_ids else resolve_model(client, model)
        if user:
            user_id = user if use_ids else resolve_user(client, user)
            resp = client.delete(f"/admin/models/{model_id}/assign/{user_id}")
            if resp.status_code == 204:
                result["user"] = {"status": "removed", "id": user_id}
                if not json_mode():
                    console.print("[green]User assignment removed.[/green]")
            else:
                handle_response(resp)
        if group:
            group_id = group if use_ids else resolve_group(client, group)
            resp = client.delete(f"/admin/models/{model_id}/assign-group/{group_id}")
            if resp.status_code == 204:
                result["group"] = {"status": "removed", "id": group_id}
                if not json_mode():
                    console.print("[green]Group assignment removed.[/green]")
            else:
                handle_response(resp)
    if json_mode():
        print_json(result)
