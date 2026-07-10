from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ttllm.api.messages import _handle_streaming
from ttllm.schemas.anthropic import MessagesRequest, MessagesResponse, TextBlock, Usage


class FakeStreamState:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
        self.latency_ms = 0

    def get_cost(self) -> Decimal:
        return Decimal("0.42")

    def get_metadata(self) -> dict:
        return {
            "provider": "bedrock",
            "latency_ms": self.latency_ms,
            "cost": {"total": str(self.get_cost())},
        }

    def get_response(self) -> MessagesResponse:
        return MessagesResponse(
            id="msg_disconnect",
            model="claude-test",
            content=[TextBlock(text="completed upstream")],
            usage=Usage(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


@pytest.mark.asyncio
async def test_stream_disconnect_drains_provider_and_logs_final_cost(monkeypatch):
    state = FakeStreamState()
    metadata_ready = asyncio.Event()
    audit_rows: list[dict] = []

    async def fake_sse_stream():
        yield "event: message_start\ndata: {}\n\n"
        await metadata_ready.wait()
        state.input_tokens = 100
        state.output_tokens = 20
        state.cache_read_tokens = 10
        yield "event: message_delta\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    def fake_gateway_stream(body, llm_model, request_id):
        return state, fake_sse_stream()

    async def fake_log_request(db, **kwargs):
        audit_rows.append(kwargs)
        return SimpleNamespace(id=uuid.uuid4(), **kwargs)

    monkeypatch.setattr("ttllm.api.messages.gateway.stream", fake_gateway_stream)
    monkeypatch.setattr("ttllm.api.messages.audit_service.log_request", fake_log_request)

    body = MessagesRequest(
        model="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
        stream=True,
    )
    model = SimpleNamespace(id=uuid.uuid4())
    user = SimpleNamespace(id=uuid.uuid4())
    response = await _handle_streaming(body, model, user, object(), uuid.uuid4(), {})

    iterator = response.body_iterator
    assert await iterator.__anext__() == "event: message_start\ndata: {}\n\n"

    pending_read = asyncio.create_task(iterator.__anext__())
    await asyncio.sleep(0)
    pending_read.cancel()
    metadata_ready.set()

    with pytest.raises(asyncio.CancelledError):
        await pending_read

    assert audit_rows
    row = audit_rows[0]
    assert row["status_code"] == 499
    assert row["error_message"] == "Client disconnected during streaming response"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 20
    assert row["total_cost"] == "0.42"
    assert row["provider_metadata"]["provider"] == "bedrock"


@pytest.mark.asyncio
async def test_stream_provider_error_records_audit_row(monkeypatch):
    """Bedrock's error branch sets state.error and yields an error frame; the audit row
    should reflect a failure (non-200 status, error_message populated) rather than a
    silent 'successful call, 0 tokens'."""
    state = FakeStreamState()
    state.error = RuntimeError("Unable to locate credentials")
    audit_rows: list[dict] = []

    async def fake_sse_stream():
        yield 'event: error\ndata: {"type": "error", "error": {"type": "api_error", "message": "Unable to locate credentials"}}\n\n'

    def fake_gateway_stream(body, llm_model, request_id):
        return state, fake_sse_stream()

    async def fake_log_request(db, **kwargs):
        audit_rows.append(kwargs)
        return SimpleNamespace(id=uuid.uuid4(), **kwargs)

    monkeypatch.setattr("ttllm.api.messages.gateway.stream", fake_gateway_stream)
    monkeypatch.setattr("ttllm.api.messages.audit_service.log_request", fake_log_request)

    body = MessagesRequest(
        model="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
        stream=True,
    )
    model = SimpleNamespace(id=uuid.uuid4())
    user = SimpleNamespace(id=uuid.uuid4())
    response = await _handle_streaming(body, model, user, object(), uuid.uuid4(), {})

    # Drain the response stream.
    frames = [chunk async for chunk in response.body_iterator]
    assert any("event: error" in f for f in frames)

    assert audit_rows, "audit row must be written for stream errors"
    row = audit_rows[0]
    assert row["status_code"] != 200
    assert row["error_message"] and "credentials" in row["error_message"].lower()


@pytest.mark.asyncio
async def test_stream_generator_raises_records_audit_row(monkeypatch):
    """If the async generator itself raises (past bedrock's own handlers), the producer
    should log, synthesize an error frame, and still write an audit row."""
    state = FakeStreamState()
    audit_rows: list[dict] = []

    async def fake_sse_stream():
        yield "event: message_start\ndata: {}\n\n"
        raise RuntimeError("upstream exploded")

    def fake_gateway_stream(body, llm_model, request_id):
        return state, fake_sse_stream()

    async def fake_log_request(db, **kwargs):
        audit_rows.append(kwargs)
        return SimpleNamespace(id=uuid.uuid4(), **kwargs)

    monkeypatch.setattr("ttllm.api.messages.gateway.stream", fake_gateway_stream)
    monkeypatch.setattr("ttllm.api.messages.audit_service.log_request", fake_log_request)

    body = MessagesRequest(
        model="claude-test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
        stream=True,
    )
    model = SimpleNamespace(id=uuid.uuid4())
    user = SimpleNamespace(id=uuid.uuid4())
    response = await _handle_streaming(body, model, user, object(), uuid.uuid4(), {})

    frames = [chunk async for chunk in response.body_iterator]
    assert any("event: error" in f and "upstream exploded" in f for f in frames)

    assert audit_rows
    row = audit_rows[0]
    assert row["status_code"] != 200
    assert row["error_message"] and "upstream exploded" in row["error_message"]
