"""CLI commands for usage and cost reporting."""

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

app = typer.Typer(help="View usage and costs")


@app.command("summary")
@app.callback(invoke_without_command=True)
def usage_summary(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO)"),
    as_json: bool = JSON_OPTION,
):
    """View usage summary."""
    params = {}
    if user_id:
        params["user_id"] = user_id
    if model_id:
        params["model_id"] = model_id
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    with get_client() as client:
        data = handle_response(client.get("/admin/usage", params=params))

    if as_json:
        print_json(data)
        return

    console.print("[bold]Usage Summary[/bold]")
    console.print(f"  Total requests: {data['total_requests']}")
    console.print(f"  Total input tokens: {data['total_input_tokens']}")
    console.print(f"  Total output tokens: {data['total_output_tokens']}")
    console.print(f"  Avg latency: {data['avg_latency_ms']}ms")


@app.command("costs")
def usage_costs(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO)"),
    as_json: bool = JSON_OPTION,
):
    """View cost breakdown by model."""
    params = {}
    if user_id:
        params["user_id"] = user_id
    if model_id:
        params["model_id"] = model_id
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    with get_client() as client:
        data = handle_response(client.get("/admin/usage/costs", params=params))

    if as_json:
        print_json(data)
        return

    table = Table(title="Cost Breakdown")
    table.add_column("Model")
    table.add_column("Requests", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Cost", justify="right")

    for item in data:
        table.add_row(
            item["model_name"],
            str(item["request_count"]),
            str(item["input_tokens"]),
            str(item["output_tokens"]),
            f"${item['total_cost']}",
        )
    console.print(table)
