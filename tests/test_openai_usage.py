"""The OpenAI-compatible client must surface usage, or tokens, cost and quotas all read zero.

Runs the real client against a local fake server, for both invoke and streaming, because the
previous client silently reported no usage while every mocked test still passed.
"""

from __future__ import annotations

import json
import threading
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from ttllm.core.providers.langchain import registry as registry_module
from ttllm.core.providers.langchain.provider import LangChainState, _read_token_counts, stream_chunks


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        usage = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
        base = {"id": "x", "created": 1, "model": "m"}
        if not body.get("stream"):
            payload = json.dumps(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        # A real server only sends usage to clients that ask for it.
        if (body.get("stream_options") or {}).get("include_usage"):
            frames.append({"choices": [], "usage": usage})
        for frame in frames:
            data = json.dumps({**base, "object": "chat.completion.chunk", **frame})
            self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def chat_model(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The SSRF guard rightly refuses loopback addresses; this server is the test's own.
    monkeypatch.setattr(registry_module, "_validate_base_url", lambda url: None)
    llm_model = SimpleNamespace(
        provider_model_id="m",
        config_json={"api_key": "k", "base_url": f"http://127.0.0.1:{server.server_port}/v1"},
    )
    yield registry_module._openai_factory(llm_model, {"max_tokens": 16})
    server.shutdown()
    server.server_close()


@pytest.mark.asyncio
async def test_invoke_reports_usage(chat_model):
    assert _read_token_counts(await chat_model.ainvoke("hello")) == (11, 7)


@pytest.mark.asyncio
async def test_stream_reports_usage_into_state(chat_model):
    llm_model = SimpleNamespace(input_cost_per_1k=Decimal("1"), output_cost_per_1k=Decimal("2"))
    state = LangChainState(llm_model, request_id=None)
    async for _ in stream_chunks(chat_model.astream("hello"), state=state):
        pass
    assert (state.input_tokens, state.output_tokens) == (11, 7)
    assert state.get_total_tokens() == 18
    assert state.get_cost() == Decimal("0.025")
