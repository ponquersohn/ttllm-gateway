"""Provider abstraction: singleton providers + per-request state.

A **provider** is a long-lived, stateless singleton that owns the expensive shared
machinery for talking to one backend (boto3 clients, thread pools, LangChain model
registries). It holds no per-request state.

A **state** (``ProviderState``) is a short-lived, per-request data bag that accumulates
everything about one request/response exchange — tokens, raw provider payload, and the
assembled response content. Crucially the state also *interprets itself*: it computes its
own cost and metadata from its own fields. The gateway treats the state as opaque, only
calling its reader methods (``get_cost`` / ``get_metadata`` / ``get_response``) and reading
``input_tokens`` / ``output_tokens``.

Cost shape is provider-specific (Bedrock bills input + output + cache read/write; other
providers may bill server-side tools, image units, etc.), so there is deliberately no
shared cost helper and no normalized usage model here — each state owns its own formula.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, AsyncIterator

from ttllm.schemas.anthropic import MessagesRequest, MessagesResponse


class ProviderState(ABC):
    """Per-request accumulator. A dumb data bag that knows how to read itself.

    May carry more than it exposes (raw payloads, partial blocks). The getters are pure
    functions of ``self`` — no network, no DB, no call back into the provider. By the time
    any getter runs, all provider activity is finished (invoke = one shot; stream = drained
    to the last chunk), so the state already contains everything it needs. Pricing lives on
    the ``llm_model`` the provider seeds in at creation, so ``get_cost`` needs nothing
    external.
    """

    input_tokens: int
    output_tokens: int
    error: BaseException | None

    @abstractmethod
    def get_cost(self) -> Decimal:
        """Total cost for this exchange, computed from this state's own fields."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Opaque provider blob persisted to ``audit_logs.provider_metadata``.

        Holds the raw provider payload, the cost breakdown, latency, stop reason, etc.
        """

    @abstractmethod
    def get_response(self) -> MessagesResponse:
        """The full Anthropic-format response for this exchange.

        For streaming this is rebuilt from the accumulated chunks; for non-streaming it is
        the single parsed response. Same shape either way.
        """


class BaseProvider(ABC):
    """Long-lived singleton. Owns connections/registries; holds NO request state.

    ``invoke`` runs a non-streaming request and returns the fully populated state.
    ``stream`` returns ``(state, sse)``: the state fills as the caller drains the SSE
    iterator, and is read (via its getters) after the stream is exhausted.
    """

    @abstractmethod
    async def invoke(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> ProviderState:
        """Execute a non-streaming request and return the filled state."""

    @abstractmethod
    def stream(
        self, request: MessagesRequest, llm_model: Any, request_id: uuid.UUID
    ) -> tuple[ProviderState, AsyncIterator[str]]:
        """Start a streaming request.

        Returns ``(state, sse_iterator)``. The state is empty until the caller drains the
        iterator, after which its getters can be read.
        """
