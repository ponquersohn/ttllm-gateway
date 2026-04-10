"""CLI commands for audit log viewing."""

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

app = typer.Typer(help="View audit logs")


@app.callback(invoke_without_command=True)
def audit_logs_list(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    limit: int = typer.Option(20, help="Number of entries"),
    as_json: bool = JSON_OPTION,
):
    """View recent audit logs."""
    params = {"limit": limit}
    if user_id:
        params["user_id"] = user_id
    if model_id:
        params["model_id"] = model_id

    with get_client() as client:
        data = handle_response(client.get("/admin/audit-logs", params=params))

    if as_json:
        print_json(data)
        return

    table = Table(title="Audit Logs")
    table.add_column("Time")
    table.add_column("Request ID", style="dim")
    table.add_column("User", style="dim")
    table.add_column("Status")
    table.add_column("Input", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Latency", justify="right")

    for log in data["items"]:
        status_style = "green" if log["status_code"] == 200 else "red"
        table.add_row(
            log["created_at"][:19],
            log["request_id"][:8] + "...",
            log["user_id"][:8] + "...",
            f"[{status_style}]{log['status_code']}[/{status_style}]",
            str(log["input_tokens"]),
            str(log["output_tokens"]),
            log.get("total_cost") or "N/A",
            f"{log['latency_ms']}ms",
        )
    console.print(table)
    console.print(f"Total: {data['total']}")
