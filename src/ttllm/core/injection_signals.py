"""Lightweight prompt-injection / jailbreak signal detector for THREAT HUNTING.
Not a security control — a telemetry signal. Low precision by design; the SIEM
correlates these with other events. Keep patterns cheap (no regex backtracking)."""
from __future__ import annotations

from ttllm.schemas.anthropic import MessagesRequest

_INJECTION_MARKERS = (
    "ignore previous instructions", "ignore all previous", "disregard the above",
    "you are now", "developer mode", "do anything now", "dan mode",
    "repeat everything above", "print your system prompt", "reveal your instructions",
)


def scan_injection_signals(request: MessagesRequest) -> list[str]:
    """Return a list of triggered signal names. Empty list = nothing flagged."""
    signals: list[str] = []
    haystack_parts: list[str] = []

    if isinstance(request.system, str):
        haystack_parts.append(request.system.lower())
    elif request.system:
        haystack_parts.extend(b.text.lower() for b in request.system if hasattr(b, "text"))

    for msg in request.messages:
        if isinstance(msg.content, str):
            haystack_parts.append(msg.content.lower())
        elif isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, dict):
                    haystack_parts.append(b.get("text", "").lower())
                elif hasattr(b, "text"):
                    haystack_parts.append((b.text or "").lower())

    haystack = " ".join(haystack_parts)
    if any(m in haystack for m in _INJECTION_MARKERS):
        signals.append("injection")
    if "system prompt" in haystack or "your instructions" in haystack:
        signals.append("prompt_extraction")
    return signals
