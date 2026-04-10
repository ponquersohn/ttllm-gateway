"""CLI command for sending test messages through the gateway."""

from __future__ import annotations

import json
from typing import Optional

import httpx
import typer

from ttllm.cli._common import console

app = typer.Typer()


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
