"""Main gateway orchestrator. Selects a provider and returns its per-request state.

Cost, metadata, and response assembly are owned by the provider's ``ProviderState`` (see
``ttllm.core.providers``). The gateway only dispatches: it picks the singleton provider for
the model and hands back the state, which the API layer reads (``get_cost`` /
``get_metadata`` / ``get_response``) when it writes the audit row.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from ttllm.core.providers import ProviderState, get_provider
from ttllm.schemas.anthropic import MessagesRequest, ServerToolDefinition


class ServerToolError(Exception):
    """Raised when a request contains server-side tools that cannot be proxied."""

    pass


def _has_server_tools(request: MessagesRequest) -> bool:
    if not request.tools:
        return False
    return any(isinstance(t, ServerToolDefinition) for t in request.tools)


def _check_server_tools(request: MessagesRequest) -> None:
    if _has_server_tools(request):
        raise ServerToolError(
            "Server-side tools (web_search, code_execution) cannot be proxied through the gateway. "
            "Remove server tool definitions and handle them client-side."
        )


async def invoke(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> ProviderState:
    """Execute a non-streaming LLM request and return the filled provider state."""
    _check_server_tools(request)
    provider = get_provider(llm_model)
    return await provider.invoke(request, llm_model, request_id)


def stream(
    request: MessagesRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> tuple[ProviderState, AsyncIterator[str]]:
    """Start a streaming LLM request.

    Returns ``(state, sse_iterator)``. The state fills as the caller drains the iterator,
    and its getters can be read once the stream is exhausted.
    """
    _check_server_tools(request)
    provider = get_provider(llm_model)
    return provider.stream(request, llm_model, request_id)
