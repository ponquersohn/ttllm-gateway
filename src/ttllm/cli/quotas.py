"""CLI commands for quota management."""

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
    resolve_quota,
)

app = TtllmTyper(help="Manage quotas")


def _describe_window(q: dict) -> str:
    if q["window_type"] == "calendar":
        return f"{q['window_unit']} ({q['timezone']})"
    return f"session {q['window_duration_seconds']}s"


@app.command("list")
def quotas_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
):
    """List all quotas."""
    with get_client() as client:
        data = handle_response(client.get("/admin/quotas", params={"offset": offset, "limit": limit}))

    if json_mode():
        print_json(data)
        return

    table = Table(title="Quotas")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Measure")
    table.add_column("Limit")
    table.add_column("Window")
    table.add_column("Dimensions")
    table.add_column("Enforcement")
    table.add_column("Enabled")

    for q in data["items"]:
        table.add_row(
            q["id"][:8] + "...",
            q["name"],
            q["measure"],
            q["limit_value"],
            _describe_window(q),
            ", ".join(q["dimensions"]) or "(shared)",
            q["enforcement"],
            "Yes" if q["enabled"] else "No",
        )
    console.print(table)


@app.command("show")
def quotas_show(
    quota: str = typer.Argument(help="Quota name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Show quota details."""
    with get_client() as client:
        quota_id = quota if use_ids else resolve_quota(client, quota)
        data = handle_response(client.get(f"/admin/quotas/{quota_id}"))

    if json_mode():
        print_json(data)
        return

    console.print(f"[bold]Quota:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Measure: {data['measure']}")
    console.print(f"  Limit: {data['limit_value']}")
    console.print(f"  Window: {_describe_window(data)}")
    console.print(f"  Dimensions: {', '.join(data['dimensions']) or '(shared)'}")
    console.print(f"  Enforcement: {data['enforcement']}")
    console.print(f"  Enabled: {'Yes' if data['enabled'] else 'No'}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  On exceeded: {json.dumps(data['on_exceeded'], indent=2)}")


@app.command("create")
def quotas_create(
    name: str = typer.Option(..., help="Quota name"),
    measure: str = typer.Option(..., help="requests, tokens, or cost"),
    limit: str = typer.Option(..., "--limit", help="Limit per window"),
    window: Optional[str] = typer.Option(None, help="Calendar window: hour, day, week, or month"),
    session_seconds: Optional[int] = typer.Option(
        None, "--session-seconds", help="Session window length; starts at the first request"
    ),
    dimension: Optional[list[str]] = typer.Option(
        None, "--dimension", help="user, model, or token (repeatable; default: user)"
    ),
    shared: bool = typer.Option(False, "--shared", help="No dimensions: one bucket shared by all matching requests"),
    timezone: str = typer.Option("UTC", help="Timezone for calendar windows"),
    enforcement: str = typer.Option("soft", help="soft or strict"),
    on_exceeded: Optional[str] = typer.Option(None, "--on-exceeded", help="JSON on_exceeded object"),
    description: Optional[str] = typer.Option(None, help="Quota description"),
    disabled: bool = typer.Option(False, "--disabled", help="Create quota as disabled"),
):
    """Create a new quota. It applies once a rule attaches it (see `rules create --apply-quota`)."""
    if bool(window) == bool(session_seconds):
        console.print("[red]Provide exactly one of --window or --session-seconds.[/red]")
        raise typer.Exit(1)
    if shared and dimension:
        console.print("[red]--shared can't be combined with --dimension.[/red]")
        raise typer.Exit(1)
    body: dict = {
        "name": name,
        "measure": measure,
        "limit_value": limit,
        "window_type": "calendar" if window else "session",
        "window_unit": window,
        "window_duration_seconds": session_seconds,
        "timezone": timezone,
        "enforcement": enforcement,
        "dimensions": [] if shared else (dimension or ["user"]),
        "enabled": not disabled,
    }
    if on_exceeded is not None:
        body["on_exceeded"] = json.loads(on_exceeded)
    if description is not None:
        body["description"] = description
    with get_client() as client:
        data = handle_response(client.post("/admin/quotas", json=body))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Quota created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Limit: {data['limit_value']} {data['measure']} per {_describe_window(data)}")


@app.command("update")
def quotas_update(
    quota: str = typer.Argument(help="Quota name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New name"),
    limit: Optional[str] = typer.Option(None, "--limit", help="New limit; applies immediately"),
    enforcement: Optional[str] = typer.Option(None, "--enforcement", help="soft or strict"),
    on_exceeded: Optional[str] = typer.Option(None, "--on-exceeded", help="New JSON on_exceeded object"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    enabled: Optional[bool] = typer.Option(None, "--enabled", help="Enable/disable quota"),
    fields: Optional[str] = typer.Option(
        None, "--fields", help="JSON of other fields (measure, window, dimensions); changing these resets its buckets"
    ),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Update an existing quota."""
    body: dict = json.loads(fields) if fields else {}
    if name is not None:
        body["name"] = name
    if limit is not None:
        body["limit_value"] = limit
    if enforcement is not None:
        body["enforcement"] = enforcement
    if on_exceeded is not None:
        body["on_exceeded"] = json.loads(on_exceeded)
    if description is not None:
        body["description"] = description
    if enabled is not None:
        body["enabled"] = enabled
    if not body:
        console.print("[red]Nothing to update. Provide at least one option.[/red]")
        raise typer.Exit(1)
    with get_client() as client:
        quota_id = quota if use_ids else resolve_quota(client, quota)
        data = handle_response(client.patch(f"/admin/quotas/{quota_id}", json=body))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Quota updated:[/green] {data['name']}")


@app.command("delete")
def quotas_delete(
    quota: str = typer.Argument(help="Quota name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Delete a quota. Refused while a rule still attaches it."""
    with get_client() as client:
        quota_id = quota if use_ids else resolve_quota(client, quota)
        resp = client.delete(f"/admin/quotas/{quota_id}")
        if resp.status_code == 204:
            if json_mode():
                print_json({"status": "deleted", "id": quota_id})
            else:
                console.print("[green]Quota deleted.[/green]")
        else:
            handle_response(resp)


@app.command("buckets")
def quotas_buckets(
    quota: str = typer.Argument(help="Quota name (or ID with --use-ids)"),
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Show a quota's current usage per bucket."""
    with get_client() as client:
        quota_id = quota if use_ids else resolve_quota(client, quota)
        data = handle_response(
            client.get(f"/admin/quotas/{quota_id}/buckets", params={"offset": offset, "limit": limit})
        )

    if json_mode():
        print_json(data)
        return

    table = Table(title=f"Buckets ({data['total']})")
    table.add_column("Dimensions")
    table.add_column("Used")
    table.add_column("In flight")
    table.add_column("Limit")
    table.add_column("Window ends")

    for b in data["items"]:
        values = b["dimension_values"]
        label = ", ".join(
            v for k, v in values.items() if k in ("user_email", "model_name", "token_id")
        ) or "(shared)"
        table.add_row(label, b["used"], b["in_flight"], b["limit"], b["window_end"] or "(idle)")
    console.print(table)


@app.command("reset")
def quotas_reset(
    quota: str = typer.Argument(help="Quota name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Discard all of a quota's usage, for every bucket."""
    with get_client() as client:
        quota_id = quota if use_ids else resolve_quota(client, quota)
        data = handle_response(client.post(f"/admin/quotas/{quota_id}/reset"))
    if json_mode():
        print_json(data)
        return
    console.print(f"[green]Quota reset:[/green] {data['buckets_reset']} bucket(s) cleared")
