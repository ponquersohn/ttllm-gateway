"""Tests for provider-error -> HTTP status/type classification in the messages endpoint."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from ttllm.core.errors import ServerToolError, UnsupportedContentError
from ttllm.api.messages import _classify_provider_error, _handle_invoke


class TestClassifyProviderError:
    def test_server_tool_error_is_501(self):
        status, error_type, _ = _classify_provider_error(ServerToolError("nope"))
        assert status == 501
        assert error_type == "not_implemented_error"

    def test_unsupported_content_error_is_400(self):
        status, error_type, message = _classify_provider_error(
            UnsupportedContentError("Document content is not supported by the OpenAI-compatible provider.")
        )
        assert status == 400
        assert error_type == "invalid_request_error"
        assert "Document" in message

    def test_unknown_exception_is_500(self):
        status, error_type, _ = _classify_provider_error(RuntimeError("boom"))
        assert status == 500
        assert error_type == "api_error"


@pytest.mark.asyncio
async def test_unsupported_content_surfaces_as_400_from_handle_invoke(monkeypatch):
    """A document sent against an OpenAI-compatible model should come back as a clean
    400, not an unhandled 500 -- request_to_internal/to_langchain_messages raise
    UnsupportedContentError before any network call, and _handle_invoke must classify it."""
    from ttllm.schemas.anthropic import DocumentBlock, DocumentSource, Message, MessagesRequest

    async def fake_log_request(db, **kwargs):
        return SimpleNamespace(id=uuid.uuid4(), **kwargs)

    monkeypatch.setattr("ttllm.api.messages.audit_service.log_request", fake_log_request)

    def fake_gateway_invoke(request, llm_model, request_id):
        raise UnsupportedContentError("Document content is not supported by the OpenAI-compatible provider.")

    monkeypatch.setattr("ttllm.api.messages.gateway.invoke", fake_gateway_invoke)

    body = MessagesRequest(
        model="gpt-4o",
        messages=[
            Message(
                role="user",
                content=[DocumentBlock(source=DocumentSource(media_type="application/pdf", data="AA=="))],
            )
        ],
        max_tokens=64,
    )
    model = SimpleNamespace(
        id=uuid.uuid4(), name="gpt-4o", provider_model_id="gpt-4o", config_json={}
    )
    user = SimpleNamespace(id=uuid.uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await _handle_invoke(body, model, user, object(), uuid.uuid4(), {})

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["type"] == "invalid_request_error"
