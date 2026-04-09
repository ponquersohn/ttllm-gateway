"""CLI wrapper around the TTLLM admin API."""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="ttllm", help="TTLLM Gateway CLI")
console = Console()

JSON_OPTION = typer.Option(False, "--json", help="Output raw JSON")


def _print_json(data) -> None:
    """Print data as formatted JSON and exit."""
    console.print(json.dumps(data, indent=2, default=str), highlight=False)

users_app = typer.Typer(help="Manage users")
models_app = typer.Typer(help="Manage models")
groups_app = typer.Typer(help="Manage groups")
tokens_app = typer.Typer(help="Manage tokens")
secrets_app = typer.Typer(help="Manage secrets")
usage_app = typer.Typer(help="View usage and costs")
audit_app = typer.Typer(help="View audit logs")

app.add_typer(users_app, name="users")
app.add_typer(models_app, name="models")
app.add_typer(groups_app, name="groups")
app.add_typer(tokens_app, name="tokens")
app.add_typer(secrets_app, name="secrets")
app.add_typer(usage_app, name="usage")
app.add_typer(audit_app, name="audit-logs")

SESSION_DIR = Path.home() / ".config" / "ttllm"
SESSION_FILE = SESSION_DIR / "session.json"


def _load_session() -> dict | None:
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return None


def _save_session(data: dict) -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(data, indent=2))


def _clear_session() -> None:
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()


def _base_url() -> str:
    session = _load_session()
    if session and session.get("base_url"):
        return session["base_url"]
    return os.environ.get("TTLLM_URL", "http://localhost:4000")


def _client() -> httpx.Client:
    base_url = _base_url()
    session = _load_session()
    headers = {}

    if session and session.get("access_token"):
        headers["Authorization"] = f"Bearer {session['access_token']}"
    else:
        console.print("[red]Not logged in. Run 'ttllm login' first.[/red]")
        raise typer.Exit(1)

    return httpx.Client(base_url=base_url, headers=headers, timeout=30)


def _handle_response(resp: httpx.Response) -> dict:
    if resp.status_code == 401:
        # Try refresh
        session = _load_session()
        if session and session.get("refresh_token"):
            refresh_resp = httpx.post(
                f"{_base_url()}/auth/token/refresh",
                json={"refresh_token": session["refresh_token"]},
                timeout=10,
            )
            if refresh_resp.status_code == 200:
                data = refresh_resp.json()
                session["access_token"] = data["access_token"]
                session["refresh_token"] = data.get("refresh_token", session["refresh_token"])
                _save_session(session)
                console.print("[yellow]Session refreshed. Please retry your command.[/yellow]")
                raise typer.Exit(0)
        console.print("[red]Session expired. Run 'ttllm login' again.[/red]")
        raise typer.Exit(1)
    if resp.status_code >= 400:
        console.print(f"[red]Error {resp.status_code}:[/red] {resp.text}")
        raise typer.Exit(1)
    return resp.json()


# --- Name resolution helpers ---


def _resolve_user(client: httpx.Client, name: str) -> str:
    """Resolve a user name to a user ID."""
    data = _handle_response(client.get("/admin/users", params={"limit": 200}))
    needle = name.lower()
    for u in data["items"]:
        if u["name"].lower() == needle or u["email"].lower() == needle:
            return u["id"]
    console.print(f"[red]User not found: {name}[/red]")
    raise typer.Exit(1)


def _resolve_group(client: httpx.Client, name: str) -> str:
    """Resolve a group name to a group ID."""
    data = _handle_response(client.get("/admin/groups", params={"limit": 200}))
    needle = name.lower()
    for g in data["items"]:
        if g["name"].lower() == needle:
            return g["id"]
    console.print(f"[red]Group not found: {name}[/red]")
    raise typer.Exit(1)


def _resolve_model(client: httpx.Client, name: str) -> str:
    """Resolve a model name to a model ID."""
    data = _handle_response(client.get("/admin/models", params={"limit": 200}))
    needle = name.lower()
    for m in data["items"]:
        if m["name"].lower() == needle:
            return m["id"]
    console.print(f"[red]Model not found: {name}[/red]")
    raise typer.Exit(1)


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

    resp = httpx.post(
        f"{base_url}/auth/token",
        json={"email": email, "password": password},
        timeout=10,
    )
    if resp.status_code != 200:
        console.print(f"[red]Login failed:[/red] {resp.text}")
        raise typer.Exit(1)

    data = resp.json()
    _save_session({
        "access_token": data["access_token"],
        "refresh_token": data.get("refresh_token"),
        "base_url": base_url,
    })
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
        _save_session({
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "base_url": base_url,
        })
        console.print("[green]SSO login successful.[/green]")
    else:
        console.print("[red]SSO login failed: no token received.[/red]")
        raise typer.Exit(1)


@app.command("logout")
def logout():
    """Log out: revoke refresh token and clear local session."""
    session = _load_session()
    if session and session.get("refresh_token") and session.get("access_token"):
        try:
            httpx.post(
                f"{_base_url()}/auth/logout",
                json={"refresh_token": session["refresh_token"]},
                headers={"Authorization": f"Bearer {session['access_token']}"},
                timeout=10,
            )
        except Exception:
            pass
    _clear_session()
    console.print("[green]Logged out.[/green]")


@app.command("whoami")
def whoami(
    as_json: bool = JSON_OPTION,
):
    """Show current user, groups, and permissions."""
    with _client() as client:
        data = _handle_response(client.get("/admin/me"))

    if as_json:
        _print_json(data)
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
    """Show server version and status."""
    with _client() as client:
        data = _handle_response(client.get("/admin/status"))
    if as_json:
        _print_json(data)
        return
    console.print(f"[bold]Version:[/bold] {data['version']}")
    console.print(f"[bold]Status:[/bold]  {data['status']}")


# --- Users ---


@users_app.command("list")
def users_list(
    offset: int = typer.Option(0, help="Offset for pagination"),
    limit: int = typer.Option(50, help="Limit for pagination"),
    as_json: bool = JSON_OPTION,
):
    """List all users."""
    with _client() as client:
        data = _handle_response(
            client.get("/admin/users", params={"offset": offset, "limit": limit})
        )

    if as_json:
        _print_json(data)
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


@users_app.command("create")
def users_create(
    name: str = typer.Option(..., help="User name"),
    email: str = typer.Option(..., help="User email"),
    password: Optional[str] = typer.Option(None, help="Password (for internal users)"),
):
    """Create a new user."""
    body = {"name": name, "email": email}
    if password:
        body["password"] = password
    with _client() as client:
        data = _handle_response(client.post("/admin/users", json=body))
    console.print(f"[green]User created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")
    console.print(f"  Email: {data['email']}")


@users_app.command("show")
def users_show(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show user details including groups and permissions."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        data = _handle_response(client.get(f"/admin/users/{user_id}"))

    if as_json:
        _print_json(data)
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


@users_app.command("models")
def users_models(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """List all models a user can access (direct + group assignments)."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        data = _handle_response(client.get(f"/admin/users/{user_id}/models"))

    if as_json:
        _print_json(data)
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


@users_app.command("update")
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
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        data = _handle_response(client.patch(f"/admin/users/{user_id}", json=body))
    console.print(f"[green]User updated:[/green] {data['name']}")


@users_app.command("delete")
def users_delete(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Deactivate a user."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        resp = client.delete(f"/admin/users/{user_id}")
        if resp.status_code == 204:
            console.print("[green]User deactivated.[/green]")
        else:
            _handle_response(resp)


@users_app.command("permissions")
def users_permissions(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show a user's direct and effective permissions."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        data = _handle_response(client.get(f"/admin/users/{user_id}/permissions"))

    if as_json:
        _print_json(data)
        return

    console.print("[bold]Direct permissions:[/bold]")
    for p in data.get("direct_permissions", []):
        console.print(f"  - {p}")
    if not data.get("direct_permissions"):
        console.print("  (none)")

    console.print("\n[bold]Effective permissions (direct + groups):[/bold]")
    for p in data.get("effective_permissions", []):
        console.print(f"  - {p}")


@users_app.command("add-permission")
def users_add_permission(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    permission: list[str] = typer.Option(..., "--permission", help="Permission(s) to assign"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Assign permission(s) directly to a user."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        data = _handle_response(
            client.post(f"/admin/users/{user_id}/permissions", json={"permissions": permission})
        )
    for r in data.get("permissions", []):
        console.print(f"  {r['permission']}: {r['status']}")


@users_app.command("remove-permission")
def users_remove_permission(
    user: str = typer.Argument(help="User name or email (or ID with --use-ids)"),
    permission: str = typer.Option(..., "--permission", help="Permission to remove"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Remove a direct permission from a user."""
    with _client() as client:
        user_id = user if use_ids else _resolve_user(client, user)
        resp = client.delete(f"/admin/users/{user_id}/permissions/{permission}")
        if resp.status_code == 204:
            console.print(f"[green]Permission '{permission}' removed from user.[/green]")
        else:
            _handle_response(resp)


# --- Models ---


@models_app.command("list")
def models_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    as_json: bool = JSON_OPTION,
):
    """List all models."""
    with _client() as client:
        data = _handle_response(
            client.get("/admin/models", params={"offset": offset, "limit": limit})
        )

    if as_json:
        _print_json(data)
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


@models_app.command("show")
def models_show(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show model details."""
    with _client() as client:
        model_id = model if use_ids else _resolve_model(client, model)
        data = _handle_response(client.get(f"/admin/models/{model_id}"))

    if as_json:
        _print_json(data)
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


@models_app.command("create")
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
    with _client() as client:
        data = _handle_response(
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


@models_app.command("update")
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
    with _client() as client:
        model_id = model if use_ids else _resolve_model(client, model)
        data = _handle_response(client.patch(f"/admin/models/{model_id}", json=body))
    console.print(f"[green]Model updated:[/green] {data['name']}")


@models_app.command("delete")
def models_delete(
    model: str = typer.Argument(help="Model name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat model argument as UUID"),
):
    """Deactivate a model."""
    with _client() as client:
        model_id = model if use_ids else _resolve_model(client, model)
        resp = client.delete(f"/admin/models/{model_id}")
        if resp.status_code == 204:
            console.print("[green]Model deactivated.[/green]")
        else:
            _handle_response(resp)


@models_app.command("assign")
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
    with _client() as client:
        model_id = model if use_ids else _resolve_model(client, model)
        if user:
            user_ids = user if use_ids else [_resolve_user(client, u) for u in user]
            data = _handle_response(
                client.post(
                    f"/admin/models/{model_id}/assign",
                    json={"user_ids": user_ids},
                )
            )
            for a in data.get("assignments", []):
                console.print(f"  User {a['user_id'][:8]}...: {a['status']}")
        if group:
            group_ids = group if use_ids else [_resolve_group(client, g) for g in group]
            data = _handle_response(
                client.post(
                    f"/admin/models/{model_id}/assign-group",
                    json={"group_ids": group_ids},
                )
            )
            for a in data.get("assignments", []):
                console.print(f"  Group {a['group_id'][:8]}...: {a['status']}")


@models_app.command("unassign")
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
    with _client() as client:
        model_id = model if use_ids else _resolve_model(client, model)
        if user:
            user_id = user if use_ids else _resolve_user(client, user)
            resp = client.delete(f"/admin/models/{model_id}/assign/{user_id}")
            if resp.status_code == 204:
                console.print("[green]User assignment removed.[/green]")
            else:
                _handle_response(resp)
        if group:
            group_id = group if use_ids else _resolve_group(client, group)
            resp = client.delete(f"/admin/models/{model_id}/assign-group/{group_id}")
            if resp.status_code == 204:
                console.print("[green]Group assignment removed.[/green]")
            else:
                _handle_response(resp)


# --- Groups ---


@groups_app.command("list")
def groups_list(
    offset: int = typer.Option(0),
    limit: int = typer.Option(50),
    as_json: bool = JSON_OPTION,
):
    """List all groups."""
    with _client() as client:
        data = _handle_response(
            client.get("/admin/groups", params={"offset": offset, "limit": limit})
        )

    if as_json:
        _print_json(data)
        return

    table = Table(title="Groups")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Permissions")
    table.add_column("Active")

    for g in data["items"]:
        table.add_row(
            g["id"][:8] + "...",
            g["name"],
            g.get("description") or "-",
            ", ".join(g.get("permissions", [])) or "-",
            "Yes" if g["is_active"] else "No",
        )
    console.print(table)


@groups_app.command("create")
def groups_create(
    name: str = typer.Option(..., help="Group name"),
    description: Optional[str] = typer.Option(None, help="Group description"),
):
    """Create a new group."""
    body = {"name": name}
    if description:
        body["description"] = description
    with _client() as client:
        data = _handle_response(client.post("/admin/groups", json=body))
    console.print(f"[green]Group created:[/green] {data['id']}")
    console.print(f"  Name: {data['name']}")


@groups_app.command("show")
def groups_show(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
    as_json: bool = JSON_OPTION,
):
    """Show group details."""
    with _client() as client:
        group_id = group if use_ids else _resolve_group(client, group)
        data = _handle_response(client.get(f"/admin/groups/{group_id}"))

    if as_json:
        _print_json(data)
        return

    console.print(f"[bold]Group:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:10]}")
    perms = data.get("permissions", [])
    if perms:
        console.print(f"\n[bold]Permissions:[/bold] {', '.join(perms)}")
    else:
        console.print("\n[bold]Permissions:[/bold] (none)")


@groups_app.command("update")
def groups_update(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    name: Optional[str] = typer.Option(None, "--name", help="New group name"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Update a group's name or description."""
    body: dict = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if not body:
        console.print("[red]Nothing to update. Provide --name or --description.[/red]")
        raise typer.Exit(1)
    with _client() as client:
        group_id = group if use_ids else _resolve_group(client, group)
        data = _handle_response(client.patch(f"/admin/groups/{group_id}", json=body))
    console.print(f"[green]Group updated:[/green] {data['name']}")


@groups_app.command("delete")
def groups_delete(
    group: str = typer.Argument(help="Group name (or ID with --use-ids)"),
    use_ids: bool = typer.Option(False, "--use-ids", help="Treat argument as UUID"),
):
    """Delete a group."""
    with _client() as client:
        group_id = group if use_ids else _resolve_group(client, group)
        resp = client.delete(f"/admin/groups/{group_id}")
        if resp.status_code == 204:
            console.print("[green]Group deleted.[/green]")
        else:
            _handle_response(resp)


@groups_app.command("add-permission")
def groups_add_permission(
    group_id: str = typer.Argument(help="Group ID"),
    permission: str = typer.Option(..., "--permission", help="Permission to assign"),
):
    """Assign a permission to a group."""
    with _client() as client:
        data = _handle_response(
            client.post(f"/admin/groups/{group_id}/permissions", json={"permission": permission})
        )
    console.print(f"[green]Permission '{permission}' assigned to group.[/green]")


@groups_app.command("remove-permission")
def groups_remove_permission(
    group_id: str = typer.Argument(help="Group ID"),
    permission: str = typer.Option(..., "--permission", help="Permission to remove"),
):
    """Remove a permission from a group."""
    with _client() as client:
        resp = client.delete(f"/admin/groups/{group_id}/permissions/{permission}")
        if resp.status_code == 204:
            console.print(f"[green]Permission '{permission}' removed from group.[/green]")
        else:
            _handle_response(resp)


@groups_app.command("add-member")
def groups_add_member(
    group_id: str = typer.Argument(help="Group ID"),
    user: list[str] = typer.Option(..., "--user", help="User ID(s) to add"),
):
    """Add user(s) to a group."""
    with _client() as client:
        data = _handle_response(
            client.post(f"/admin/groups/{group_id}/members", json={"user_ids": user})
        )
    for m in data.get("members", []):
        console.print(f"  User {m['user_id'][:8]}...: {m['status']}")


@groups_app.command("remove-member")
def groups_remove_member(
    group_id: str = typer.Argument(help="Group ID"),
    user: str = typer.Option(..., "--user", help="User ID to remove"),
):
    """Remove a user from a group."""
    with _client() as client:
        resp = client.delete(f"/admin/groups/{group_id}/members/{user}")
        if resp.status_code == 204:
            console.print("[green]Member removed.[/green]")
        else:
            _handle_response(resp)


# --- Tokens ---


@tokens_app.command("create")
def tokens_create(
    user_id: Optional[str] = typer.Option(None, "--user", help="User ID (defaults to current user)"),
    label: Optional[str] = typer.Option(None, "--label", help="Token label"),
    ttl_days: Optional[int] = typer.Option(None, "--ttl-days", help="Token lifetime in days (default: 30, max: 365)"),
    permissions: Optional[str] = typer.Option(None, "--permissions", help="Comma-separated permissions (default: llm.invoke)"),
):
    """Generate a token with specified permissions (default: llm.invoke for gateway access)."""
    body: dict = {}
    if user_id:
        body["user_id"] = user_id
    if label:
        body["label"] = label
    if ttl_days is not None:
        body["ttl_days"] = ttl_days
    if permissions:
        body["permissions"] = [s.strip() for s in permissions.split(",") if s.strip()]
    with _client() as client:
        data = _handle_response(client.post("/admin/tokens", json=body))
    console.print("[green]Token created:[/green]")
    console.print(f"  Token: [bold]{data['access_token']}[/bold]")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Permissions: {', '.join(data.get('permissions', []))}")
    console.print(f"  Label: {data.get('label') or 'N/A'}")
    console.print(f"  Expires: {data.get('expires_at') or 'never'}")
    console.print("[yellow]Save this token now -- it will not be shown again.[/yellow]")


@tokens_app.command("show")
def tokens_show(
    token_id: str = typer.Argument(help="Token ID"),
    as_json: bool = JSON_OPTION,
):
    """Show token details (token value is never displayed)."""
    with _client() as client:
        data = _handle_response(client.get(f"/admin/tokens/{token_id}"))

    if as_json:
        _print_json(data)
        return

    console.print(f"[bold]Token:[/bold] {data['id'][:5]}...")
    console.print(f"  ID: {data['id']}")
    console.print(f"  User: {data.get('user_email') or data['user_id']}")
    console.print(f"  Label: {data.get('label') or '(none)'}")
    console.print(f"  Permissions: {', '.join(data.get('permissions', []))}")
    console.print(f"  Active: {'Yes' if data['is_active'] else 'No'}")
    console.print(f"  Created: {data['created_at'][:19]}")
    console.print(f"  Expires: {(data.get('expires_at') or 'never')[:19]}")


@tokens_app.command("list")
def tokens_list(
    user_id: Optional[str] = typer.Option(None, "--user", help="Filter by user ID"),
    as_json: bool = JSON_OPTION,
):
    """List active tokens."""
    params = {}
    if user_id:
        params["user_id"] = user_id
    with _client() as client:
        data = _handle_response(client.get("/admin/tokens", params=params))

    if as_json:
        _print_json(data)
        return

    table = Table(title="Tokens")
    table.add_column("ID", style="dim")
    table.add_column("User ID", style="dim")
    table.add_column("Email")
    table.add_column("Label")
    table.add_column("Permissions")
    table.add_column("Active")
    table.add_column("Created")
    table.add_column("Expires")

    for t in data:
        table.add_row(
            t["id"][:8] + "...",
            t["user_id"][:8] + "...",
            t.get("user_email") or "-",
            t.get("label") or "-",
            ", ".join(t.get("permissions", [])) or "-",
            "Yes" if t["is_active"] else "No",
            t["created_at"][:10],
            (t.get("expires_at") or "never")[:10],
        )
    console.print(table)


@tokens_app.command("delete")
def tokens_delete(
    token_id: str = typer.Argument(help="Token ID to revoke"),
):
    """Revoke a token."""
    with _client() as client:
        resp = client.delete(f"/admin/tokens/{token_id}")
        if resp.status_code == 204:
            console.print("[green]Token revoked.[/green]")
        else:
            _handle_response(resp)


# --- Secrets ---


def _resolve_secret(client: httpx.Client, name: str) -> str:
    """Resolve a secret name to a secret ID."""
    data = _handle_response(client.get("/admin/secrets", params={"limit": 200}))
    needle = name.lower()
    for s in data["items"]:
        if s["name"].lower() == needle:
            return s["id"]
    console.print(f"[red]Secret not found: {name}[/red]")
    raise typer.Exit(1)


@secrets_app.command("list")
def secrets_list(
    offset: int = typer.Option(0, help="Offset for pagination"),
    limit: int = typer.Option(50, help="Limit for pagination"),
    as_json: bool = JSON_OPTION,
):
    """List all secrets (values are never shown)."""
    with _client() as client:
        data = _handle_response(
            client.get("/admin/secrets", params={"offset": offset, "limit": limit})
        )

    if as_json:
        _print_json(data)
        return

    table = Table(title="Secrets")
    table.add_column("ID", style="dim")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Created")
    table.add_column("Updated")

    for s in data["items"]:
        table.add_row(
            s["id"][:8] + "...",
            s["name"],
            s.get("description") or "-",
            s["created_at"][:10],
            s["updated_at"][:10],
        )
    console.print(table)
    console.print(f"Total: {data['total']}")


@secrets_app.command("create")
def secrets_create(
    name: str = typer.Option(..., "--name", help="Secret name"),
    value: Optional[str] = typer.Option(None, "--value", help="Secret value (if omitted, prompted with hidden input)"),
    description: Optional[str] = typer.Option(None, "--description", help="Description"),
):
    """Create a new secret. Value is prompted with hidden input."""
    if value is None:
        value = typer.prompt("Secret value", hide_input=True)
    body: dict = {"name": name, "value": value}
    if description:
        body["description"] = description
    with _client() as client:
        data = _handle_response(client.post("/admin/secrets", json=body))
    console.print(f"[green]Secret created:[/green] {data['name']}")


@secrets_app.command("show")
def secrets_show(
    name: str = typer.Argument(help="Secret name"),
    as_json: bool = JSON_OPTION,
):
    """Show secret metadata (value is never displayed)."""
    with _client() as client:
        secret_id = _resolve_secret(client, name)
        data = _handle_response(client.get(f"/admin/secrets/{secret_id}"))

    if as_json:
        _print_json(data)
        return

    console.print(f"[bold]Secret:[/bold] {data['name']}")
    console.print(f"  ID: {data['id']}")
    console.print(f"  Description: {data.get('description') or '(none)'}")
    console.print(f"  Created: {data['created_at'][:19]}")
    console.print(f"  Updated: {data['updated_at'][:19]}")


@secrets_app.command("update")
def secrets_update(
    name: str = typer.Argument(help="Secret name"),
    prompt_value: bool = typer.Option(False, "--prompt-value", help="Prompt for a new secret value"),
    description: Optional[str] = typer.Option(None, "--description", help="New description"),
):
    """Update a secret's value or description."""
    body: dict = {}
    if prompt_value:
        body["value"] = typer.prompt("New secret value", hide_input=True)
    if description is not None:
        body["description"] = description
    if not body:
        console.print("[red]Nothing to update. Provide --prompt-value or --description.[/red]")
        raise typer.Exit(1)
    with _client() as client:
        secret_id = _resolve_secret(client, name)
        data = _handle_response(client.patch(f"/admin/secrets/{secret_id}", json=body))
    console.print(f"[green]Secret updated:[/green] {data['name']}")


@secrets_app.command("delete")
def secrets_delete(
    name: str = typer.Argument(help="Secret name"),
):
    """Delete a secret."""
    with _client() as client:
        secret_id = _resolve_secret(client, name)
        resp = client.delete(f"/admin/secrets/{secret_id}")
        if resp.status_code == 204:
            console.print("[green]Secret deleted.[/green]")
        else:
            _handle_response(resp)


# --- Usage ---


@usage_app.command("summary")
@usage_app.callback(invoke_without_command=True)
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

    with _client() as client:
        data = _handle_response(client.get("/admin/usage", params=params))

    if as_json:
        _print_json(data)
        return

    console.print("[bold]Usage Summary[/bold]")
    console.print(f"  Total requests: {data['total_requests']}")
    console.print(f"  Total input tokens: {data['total_input_tokens']}")
    console.print(f"  Total output tokens: {data['total_output_tokens']}")
    console.print(f"  Avg latency: {data['avg_latency_ms']}ms")


@usage_app.command("costs")
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

    with _client() as client:
        data = _handle_response(client.get("/admin/usage/costs", params=params))

    if as_json:
        _print_json(data)
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


# --- Audit Logs ---


@audit_app.callback(invoke_without_command=True)
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

    with _client() as client:
        data = _handle_response(client.get("/admin/audit-logs", params=params))

    if as_json:
        _print_json(data)
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


# --- Chat ---


def _handle_chat_error(resp: httpx.Response, model: str, url: str) -> None:
    """Print a user-friendly error for common HTTP failures and exit."""
    status = resp.status_code
    if status == 401:
        console.print("[red]Authentication failed. Check your token.[/red]")
    elif status == 403:
        console.print(f"[red]Access denied. Your token may not have access to model '{model}'.[/red]")
    elif status == 404:
        console.print(f"[red]Endpoint not found. Check the gateway URL: {url}[/red]")
    elif status == 429:
        console.print("[red]Rate limited. Try again later.[/red]")
    else:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        console.print(f"[red]Gateway error ({status}): {detail}[/red]")
    raise typer.Exit(1)


@app.command("chat")
def chat(
    message: str = typer.Argument(..., help="Message to send to the LLM"),
    model: str = typer.Option(..., "--model", "-m", help="Model name to use"),
    token: Optional[str] = typer.Option(None, "--token", "-t", envvar="TTLLM_TOKEN", help="API token"),
    url: str = typer.Option("http://localhost:4000", "--url", envvar="TTLLM_URL", help="Gateway base URL"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Maximum tokens in response"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming"),
    show_usage: bool = typer.Option(False, "--usage", help="Show token usage after response"),
):
    """Send a test message through the gateway."""
    if not token:
        console.print("[red]No token provided. Use --token or set TTLLM_TOKEN.[/red]")
        raise typer.Exit(1)

    headers = {"x-api-key": token}
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
    }
    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=10)

    try:
        if no_stream:
            body["stream"] = False
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{url}/anthropic/v1/messages", json=body, headers=headers)
            if resp.status_code != 200:
                _handle_chat_error(resp, model, url)
            data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    console.print(block["text"], highlight=False, markup=False)
            if show_usage:
                usage = data.get("usage", {})
                console.print(f"\n[dim]Tokens: {usage.get('input_tokens', 0)} input, {usage.get('output_tokens', 0)} output[/dim]")
        else:
            body["stream"] = True
            input_tokens = 0
            output_tokens = 0
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", f"{url}/anthropic/v1/messages", json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        resp.read()
                        _handle_chat_error(resp, model, url)
                    for line in resp.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = json.loads(line[6:])
                        event_type = payload.get("type")
                        if event_type == "content_block_delta":
                            delta = payload.get("delta", {})
                            if delta.get("type") == "text_delta":
                                console.print(delta["text"], end="", highlight=False, markup=False)
                        elif event_type == "message_start":
                            msg = payload.get("message", {})
                            usage = msg.get("usage", {})
                            input_tokens = usage.get("input_tokens", 0)
                        elif event_type == "message_delta":
                            usage = payload.get("usage", {})
                            output_tokens = usage.get("output_tokens", 0)
            console.print()  # final newline
            if show_usage:
                console.print(f"[dim]Tokens: {input_tokens} input, {output_tokens} output[/dim]")
    except httpx.ConnectError:
        console.print(f"[red]Could not connect to {url}. Is the gateway running?[/red]")
        raise typer.Exit(1)
    except httpx.TimeoutException:
        console.print("[red]Request timed out.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
