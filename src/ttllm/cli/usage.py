"""CLI commands for usage and cost reporting."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
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


_RELATIVE_TIME_RE = re.compile(r"^-(\d+)([mhdw])$")
_RELATIVE_TIME_UNITS = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def _normalize_time(value: str | None) -> str | None:
    if value is None:
        return None
    match = _RELATIVE_TIME_RE.fullmatch(value)
    if not match:
        return value
    amount = int(match.group(1))
    unit = _RELATIVE_TIME_UNITS[match.group(2)]
    return (datetime.now(timezone.utc) - timedelta(**{unit: amount})).isoformat()


@app.callback(invoke_without_command=True)
def usage_callback(
    ctx: typer.Context,
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    email: Optional[str] = typer.Option(None, "--email", help="Filter by user email"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO or relative like -24h)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO or relative like -1h)"),
    as_json: bool = JSON_OPTION,
):
    """View usage summary."""
    if ctx.invoked_subcommand is not None:
        return
    _print_usage_summary(user_id, email, model_id, since, until, as_json)


@app.command("summary")
def usage_summary(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    email: Optional[str] = typer.Option(None, "--email", help="Filter by user email"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO or relative like -24h)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO or relative like -1h)"),
    as_json: bool = JSON_OPTION,
):
    """View usage summary."""
    _print_usage_summary(user_id, email, model_id, since, until, as_json)


def _print_usage_summary(
    user_id: Optional[str],
    email: Optional[str],
    model_id: Optional[str],
    since: Optional[str],
    until: Optional[str],
    as_json: bool,
) -> None:
    params = {}
    if user_id:
        params["user_id"] = user_id
    if email:
        params["email"] = email
    if model_id:
        params["model_id"] = model_id
    if since:
        params["since"] = _normalize_time(since)
    if until:
        params["until"] = _normalize_time(until)

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
    console.print(f"  Total cost: ${data.get('total_cost', '0')}")


@app.command("costs")
def usage_costs(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    email: Optional[str] = typer.Option(None, "--email", help="Filter by user email"),
    model_id: Optional[str] = typer.Option(None, "--model", help="Filter by model ID"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO or relative like -24h)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO or relative like -1h)"),
    as_json: bool = JSON_OPTION,
):
    """View cost breakdown by model."""
    params = {}
    if user_id:
        params["user_id"] = user_id
    if email:
        params["email"] = email
    if model_id:
        params["model_id"] = model_id
    if since:
        params["since"] = _normalize_time(since)
    if until:
        params["until"] = _normalize_time(until)

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


@app.command("by-user")
def usage_by_user(
    since: Optional[str] = typer.Option(None, help="Start date (ISO or relative like -24h)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO or relative like -1h)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Show only the top N users by cost"),
    as_json: bool = JSON_OPTION,
):
    """View usage and cost grouped by user, highest cost first."""
    params = {}
    if since:
        params["since"] = _normalize_time(since)
    if until:
        params["until"] = _normalize_time(until)
    if limit is not None:
        params["limit"] = limit

    with get_client() as client:
        data = handle_response(client.get("/admin/usage/by-user", params=params))

    if as_json:
        print_json(data)
        return

    table = Table(title="Usage by User")
    table.add_column("User")
    table.add_column("Email")
    table.add_column("Requests", justify="right")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Total Cost", justify="right")

    for item in data:
        table.add_row(
            item.get("user_name") or "-",
            item.get("user_email") or "-",
            str(item["request_count"]),
            str(item["input_tokens"]),
            str(item["output_tokens"]),
            f"${item['total_cost']}",
        )
    console.print(table)
