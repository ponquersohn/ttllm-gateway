"""Main gateway orchestrator. Selects a provider and returns its per-request state.

Cost, metadata, and response assembly are owned by the provider's ``ProviderState`` (see
``ttllm.core.providers``). The gateway only dispatches: it picks the singleton provider for
the model and hands back the state, which the API layer reads (``get_cost`` /
``get_metadata`` / ``get_response``) when it writes the audit row.

Operates purely on the internal representation (``ttllm.core.model``) -- it has no
knowledge of any external wire format. It also has no opinion on provider capabilities: an
``InternalRequest`` may reference server-side tools, and it's up to the selected provider's
translation code to decide whether it can proxy them (raising ``ServerToolError`` if not) --
the gateway just dispatches.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator

from ttllm.core.model import InternalChunk, InternalRequest
from ttllm.core.providers import ProviderState, get_provider


async def invoke(
    request: InternalRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> ProviderState:
    """Execute a non-streaming LLM request and return the filled provider state."""
    provider = get_provider(llm_model)
    return await provider.invoke(request, llm_model, request_id)


def stream(
    request: InternalRequest,
    llm_model: Any,
    request_id: uuid.UUID,
) -> tuple[ProviderState, AsyncIterator[InternalChunk]]:
    """Start a streaming LLM request.

    Returns ``(state, chunks)``. The state fills as the caller drains the iterator,
    and its getters can be read once the stream is exhausted.
    """
    provider = get_provider(llm_model)
    return provider.stream(request, llm_model, request_id)
