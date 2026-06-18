"""CLI commands for generating detailed usage reports (PDF/HTML)."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from ttllm.cli._common import console, get_client, handle_response

app = typer.Typer(help="Generate usage reports (preview)")


def _fetch_report_data(client, user_id: str | None, since: str | None, until: str | None) -> dict:
    """Fetch all data needed for the report from the API."""
    params: dict = {}
    if user_id:
        params["user_id"] = user_id
    if since:
        params["since"] = since
    if until:
        params["until"] = until

    summary = handle_response(client.get("/admin/usage", params=params))
    costs = handle_response(client.get("/admin/usage/costs", params=params))
    audit_params = {**params, "limit": 100}
    audit = handle_response(client.get("/admin/audit-logs", params=audit_params))

    user_info = None
    if user_id:
        try:
            user_info = handle_response(client.get(f"/admin/users/{user_id}"))
        except SystemExit:
            pass

    user_usage = []
    if not user_id:
        by_user_params: dict = {}
        if since:
            by_user_params["since"] = since
        if until:
            by_user_params["until"] = until
        rows = handle_response(client.get("/admin/usage/by-user", params=by_user_params))
        # Already ordered by cost descending; surface a numeric cost for the report's formatting.
        user_usage = [
            {
                "user_name": r.get("user_name") or "",
                "user_email": r.get("user_email") or "",
                "request_count": r["request_count"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_cost": float(r.get("total_cost") or 0),
            }
            for r in rows
            if r["request_count"] > 0
        ]

    return {
        "summary": summary,
        "costs": costs,
        "audit_logs": audit.get("items", []),
        "audit_total": audit.get("total", 0),
        "user_info": user_info,
        "user_usage": user_usage,
    }


def _build_html(data: dict, since: str | None, until: str | None) -> str:
    """Build HTML report from data."""
    summary = data["summary"]
    costs = data["costs"]
    audit_logs = data["audit_logs"]
    audit_total = data["audit_total"]
    user_info = data["user_info"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    period_parts = []
    if since:
        period_parts.append(f"From: {since}")
    if until:
        period_parts.append(f"Until: {until}")
    period_str = " | ".join(period_parts) if period_parts else "All time"

    title = "Usage Report"
    if user_info:
        title = f"Usage Report — {html.escape(user_info['name'])}"

    total_cost = sum(float(c.get("total_cost", 0)) for c in costs)

    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; color: #1a1a1a; font-size: 14px; }}
  h1 {{ color: #1a1a1a; border-bottom: 3px solid #2563eb; padding-bottom: 10px; }}
  h2 {{ color: #374151; margin-top: 30px; }}
  .meta {{ color: #6b7280; margin-bottom: 30px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
  .summary-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
  .summary-card .label {{ color: #6b7280; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .summary-card .value {{ font-size: 24px; font-weight: 700; color: #1e293b; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th {{ background: #f1f5f9; text-align: left; padding: 10px 12px; font-weight: 600; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover td {{ background: #f8fafc; }}
  .text-right {{ text-align: right; }}
  .status-ok {{ color: #16a34a; }}
  .status-err {{ color: #dc2626; }}
  .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0; color: #9ca3af; font-size: 12px; }}
  @media print {{ body {{ margin: 20px; }} }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">
  Generated: {now}<br>
  Period: {period_str}
"""
    if user_info:
        report += f"  <br>User: {html.escape(user_info['name'])} ({html.escape(user_info['email'])})\n"
    report += "</div>\n"

    # Summary cards
    report += """<h2>Summary</h2>
<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Total Requests</div>
    <div class="value">{requests:,}</div>
  </div>
  <div class="summary-card">
    <div class="label">Input Tokens</div>
    <div class="value">{input_tokens:,}</div>
  </div>
  <div class="summary-card">
    <div class="label">Output Tokens</div>
    <div class="value">{output_tokens:,}</div>
  </div>
  <div class="summary-card">
    <div class="label">Avg Latency</div>
    <div class="value">{latency}ms</div>
  </div>
  <div class="summary-card">
    <div class="label">Total Cost</div>
    <div class="value">${total_cost:.4f}</div>
  </div>
</div>
""".format(
        requests=summary["total_requests"],
        input_tokens=summary["total_input_tokens"],
        output_tokens=summary["total_output_tokens"],
        latency=summary["avg_latency_ms"],
        total_cost=total_cost,
    )

    # Cost breakdown table
    if costs:
        report += """<h2>Cost Breakdown by Model</h2>
<table>
<thead>
<tr><th>Model</th><th class="text-right">Requests</th><th class="text-right">Input Tokens</th><th class="text-right">Output Tokens</th><th class="text-right">Cost</th></tr>
</thead>
<tbody>
"""
        for item in costs:
            report += (
                f'<tr><td>{html.escape(item["model_name"])}</td>'
                f'<td class="text-right">{item["request_count"]:,}</td>'
                f'<td class="text-right">{item["input_tokens"]:,}</td>'
                f'<td class="text-right">{item["output_tokens"]:,}</td>'
                f'<td class="text-right">${item["total_cost"]}</td></tr>\n'
            )
        report += "</tbody></table>\n"

    # Per-user breakdown table
    user_usage = data.get("user_usage", [])
    if user_usage:
        report += """<h2>Usage by User</h2>
<table>
<thead>
<tr><th>User</th><th>Email</th><th class="text-right">Requests</th><th class="text-right">Input Tokens</th><th class="text-right">Output Tokens</th><th class="text-right">Cost</th></tr>
</thead>
<tbody>
"""
        for u in sorted(user_usage, key=lambda x: x["total_cost"], reverse=True):
            report += (
                f'<tr><td>{html.escape(u["user_name"])}</td>'
                f'<td>{html.escape(u["user_email"])}</td>'
                f'<td class="text-right">{u["request_count"]:,}</td>'
                f'<td class="text-right">{u["input_tokens"]:,}</td>'
                f'<td class="text-right">{u["output_tokens"]:,}</td>'
                f'<td class="text-right">${u["total_cost"]:.4f}</td></tr>\n'
            )
        report += "</tbody></table>\n"

    # Audit log table
    if audit_logs:
        shown = len(audit_logs)
        report += f"<h2>Recent Requests ({shown} of {audit_total:,})</h2>\n"
        report += """<table>
<thead>
<tr><th>Time</th><th>Request ID</th><th>Status</th><th class="text-right">Input</th><th class="text-right">Output</th><th class="text-right">Cost</th><th class="text-right">Latency</th></tr>
</thead>
<tbody>
"""
        for log in audit_logs:
            status_class = "status-ok" if log["status_code"] == 200 else "status-err"
            cost_str = f'${log["total_cost"]}' if log.get("total_cost") else "N/A"
            report += (
                f'<tr><td>{log["created_at"][:19]}</td>'
                f'<td>{log["request_id"][:12]}...</td>'
                f'<td class="{status_class}">{log["status_code"]}</td>'
                f'<td class="text-right">{log["input_tokens"]:,}</td>'
                f'<td class="text-right">{log["output_tokens"]:,}</td>'
                f'<td class="text-right">{cost_str}</td>'
                f'<td class="text-right">{log["latency_ms"]}ms</td></tr>\n'
            )
        report += "</tbody></table>\n"

    report += f'<div class="footer">TTLLM Gateway — Usage Report generated {now}</div>\n'
    report += "</body></html>"
    return report


def _html_to_pdf(html_content: str, output_path: Path) -> None:
    """Convert HTML to PDF using fpdf2 or weasyprint if available."""
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(str(output_path))
        return
    except ImportError:
        pass

    try:
        from fpdf import FPDF, HTMLMixin

        class PDF(FPDF, HTMLMixin):
            pass

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        pdf.write_html(html_content)
        pdf.output(str(output_path))
        return
    except ImportError:
        pass

    raise ImportError(
        "PDF generation requires 'weasyprint' or 'fpdf2'. "
        "Install one with: pip install weasyprint  OR  pip install fpdf2"
    )


@app.command("generate")
def generate(
    user: Optional[str] = typer.Option(None, "--user", help="User ID or name to report on"),
    since: Optional[str] = typer.Option(None, help="Start date (ISO format)"),
    until: Optional[str] = typer.Option(None, help="End date (ISO format)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (default: report.pdf or report.html)"),
    format: str = typer.Option("pdf", "--format", "-f", help="Output format: pdf or html"),
):
    """Generate a detailed usage report for a user."""
    if format not in ("pdf", "html"):
        console.print("[red]Format must be 'pdf' or 'html'[/red]")
        raise typer.Exit(1)

    console.print("[yellow]Note: reports is a preview feature and may change in future releases.[/yellow]")

    with get_client() as client:
        user_id = user
        if user and not _is_uuid(user):
            from ttllm.cli._common import resolve_user
            user_id = resolve_user(client, user)

        console.print("[dim]Fetching report data...[/dim]")
        data = _fetch_report_data(client, user_id, since, until)

    html_content = _build_html(data, since, until)

    if not output:
        output = f"report.{format}"

    output_path = Path(output)

    if format == "html":
        output_path.write_text(html_content, encoding="utf-8")
    else:
        try:
            _html_to_pdf(html_content, output_path)
        except ImportError as e:
            console.print(f"[yellow]{e}[/yellow]")
            console.print("[dim]Falling back to HTML output...[/dim]")
            output_path = output_path.with_suffix(".html")
            output_path.write_text(html_content, encoding="utf-8")

    console.print(f"[green]Report saved to:[/green] {output_path}")


def _is_uuid(value: str) -> bool:
    """Check if a string looks like a UUID."""
    import re
    return bool(re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value, re.I))
