"""Errors shared across input adapters and providers."""

from __future__ import annotations


class UnsupportedContentError(Exception):
    """Raised when a provider (or input adapter) is asked to translate content it can
    neither represent nor safely approximate -- e.g. a document block sent to the
    OpenAI-compatible provider. Mapped to a 400 ``invalid_request_error`` at the API
    layer. See ``docs/content-mapping.md`` for the emulate-vs-error policy."""


class ServerToolError(Exception):
    """Raised by a provider when the request references server-side tools (as a tool
    definition or anywhere in message history) that it cannot proxy. Whether a provider
    can handle server-side tools is a provider-specific capability, not something the
    API layer or an input adapter should presume -- so this is raised from provider
    translation code (e.g. ``core/providers/bedrock/converse.py``,
    ``core/providers/langchain/translation.py``), not before it.
    Mapped to a 501 ``not_implemented_error`` at the API layer."""
