"""CLI wrapper around the TTLLM admin API."""

from __future__ import annotations

import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from urllib.parse import parse_qs, urlparse

import typer

from ttllm.cli._common import JSON_OPTION, console, get_client, handle_response, print_json
from ttllm.cli.client import TTLLMClient
from ttllm.cli import audit, chat, groups, me, models, secrets, tokens, usage, users

app = typer.Typer(name="ttllm", help="TTLLM Gateway CLI")

app.add_typer(users.app, name="users")
app.add_typer(models.app, name="models")
app.add_typer(groups.app, name="groups")
app.add_typer(tokens.app, name="tokens")
app.add_typer(secrets.app, name="secrets")
app.add_typer(usage.app, name="usage")
app.add_typer(audit.app, name="audit-logs")
app.add_typer(me.app, name="me")
app.command("chat")(chat.chat)


# --- Login / Logout ---


@app.command("login")
def login(
    idp: Optional[str] = typer.Option(None, "--idp", help="Identity provider slug for SSO login"),
    url: str = typer.Option("http://localhost:4000", "--url", help="TTLLM API base URL"),
):
    """Log in to TTLLM. Uses email+password by default, or --idp for SSO."""
    if idp:
        _login_sso(idp, url)
    else:
        _login_local(url)


def _login_local(base_url: str):
    email = typer.prompt("Email")
    password = typer.prompt("Password", hide_input=True)

    resp = TTLLMClient.login(base_url, email, password)
    if resp.status_code != 200:
        console.print(f"[red]Login failed:[/red] {resp.text}")
        raise typer.Exit(1)
    console.print("[green]Login successful.[/green]")


def _login_sso(idp_slug: str, base_url: str):
    """SSO login: open browser -> IdP -> API callback -> redirect to CLI with tokens."""
    import socket

    # Find a free port for the ephemeral callback server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    cli_callback = f"http://localhost:{port}/callback"
    auth_url = f"{base_url}/auth/sso/{idp_slug}/authorize?final_redirect={cli_callback}"

    token_data = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            if "access_token" in params:
                token_data["access_token"] = params["access_token"][0]
                token_data["refresh_token"] = params.get("refresh_token", [""])[0]
                token_data["received"] = True
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Login successful. You can close this window.</h2></body></html>")

        def log_message(self, format, *args):
            pass  # Suppress HTTP logs

    server = HTTPServer(("localhost", port), CallbackHandler)

    console.print("Opening browser for SSO login...")
    webbrowser.open(auth_url)
    console.print(f"Waiting for callback on localhost:{port}...")

    server.handle_request()  # Handle single request
    server.server_close()

    if token_data.get("access_token"):
        TTLLMClient.login_with_tokens(base_url, token_data["access_token"], token_data.get("refresh_token"))
        console.print("[green]SSO login successful.[/green]")
    else:
        console.print("[red]SSO login failed: no token received.[/red]")
        raise typer.Exit(1)


@app.command("logout")
def logout():
    """Log out: revoke refresh token and clear local session."""
    session = TTLLMClient.load_session()
    if session and session.get("access_token"):
        try:
            with TTLLMClient.from_session() as client:
                client.logout()
        except Exception:
            TTLLMClient.clear_session()
    else:
        TTLLMClient.clear_session()
    console.print("[green]Logged out.[/green]")


@app.command("whoami")
def whoami(
    as_json: bool = JSON_OPTION,
):
    """Show current user, groups, and permissions."""
    with get_client() as client:
        data = handle_response(client.get("/me"))

    if as_json:
        print_json(data)
        return

    console.print(f"[bold]User:[/bold] {data['name']} ({data['email']})")
    console.print(f"[bold]ID:[/bold] {data['id']}")
    console.print(f"[bold]Groups:[/bold] {', '.join(data.get('groups', [])) or '(none)'}")

    effective = set(data.get("effective_permissions", []))
    available = set(data.get("available_permissions", []))

    console.print(f"\n[bold]Effective permissions (this token):[/bold]")
    for p in sorted(effective):
        console.print(f"  - {p}")

    extra = sorted(available - effective)
    if extra:
        console.print(f"\n[bold]Available permissions (not in this token):[/bold]")
        for p in extra:
            console.print(f"  - [dim]{p}[/dim]")


@app.command("status")
def status(as_json: bool = JSON_OPTION):
    """Show server version, status, and configuration health checks."""
    with get_client() as client:
        data = handle_response(client.get("/admin/status"))
    if as_json:
        print_json(data)
        return
    console.print(f"[bold]Version:[/bold] {data['version']}")
    overall = data["status"]
    overall_style = "green" if overall == "ok" else "yellow"
    console.print(f"[bold]Status:[/bold]  [{overall_style}]{overall}[/{overall_style}]")

    checks = data.get("checks", [])
    if checks:
        console.print()
        for check in checks:
            color = {"ok": "green", "warning": "yellow", "error": "red"}.get(check["status"], "white")
            label = f"[{color}]{check['status']}[/{color}]"
            line = f"  {check['name']}: {label}"
            if check.get("message"):
                line += f" — {check['message']}"
            console.print(line)


if __name__ == "__main__":
    app()
