"""A minimal fake AWS bedrock-runtime server for integration testing.

Implements the two routes boto3 calls for the Converse API:
  POST /model/{modelId}/converse         -> JSON response
  POST /model/{modelId}/converse-stream  -> application/vnd.amazon.eventstream

It is deliberately dumb: it echoes the last user message back so tests can assert on
deterministic output, and derives token counts from text length. boto3 does not validate
response signatures, so no AWS credentials or signing are needed on this side.

Run standalone:  uvicorn tests.integration.fake_bedrock.app:app --host 0.0.0.0 --port 9099
"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response

from .eventstream_encoder import encode_converse_stream

app = FastAPI(title="fake-bedrock")


def _last_user_text(body: dict) -> str:
    """Extract a representative text from the Converse request for echoing back."""
    messages = body.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            for block in msg.get("content", []):
                if isinstance(block, dict) and "text" in block:
                    return block["text"]
    return ""


def _reply_text(body: dict) -> str:
    user_text = _last_user_text(body)
    return f"fake-bedrock echo: {user_text}" if user_text else "fake-bedrock reply"


def _token_counts(body: dict, reply: str) -> tuple[int, int]:
    # Cheap deterministic counts: ~1 token / 4 chars, min 1.
    in_chars = sum(
        len(b.get("text", ""))
        for m in body.get("messages", [])
        for b in m.get("content", [])
        if isinstance(b, dict)
    )
    return max(1, in_chars // 4), max(1, len(reply) // 4)


def _has_cache_point(body: dict) -> bool:
    """True if the request carries any Bedrock cachePoint marker.

    TTLLM inserts ``{"cachePoint": {"type": ...}}`` into the system / tools /
    message-content arrays when the client sends Anthropic ``cache_control``.
    Detecting it here lets the fake report cache-read tokens, so the integration
    suite can prove the marker survived the full translate path end-to-end.
    """
    def _scan(items: list) -> bool:
        return any(isinstance(i, dict) and "cachePoint" in i for i in items or [])

    if _scan(body.get("system", [])):
        return True
    tool_cfg = body.get("toolConfig") or {}
    if _scan(tool_cfg.get("tools", [])):
        return True
    for msg in body.get("messages", []):
        if _scan(msg.get("content", [])):
            return True
    return False


def _cache_counts(body: dict, in_tokens: int) -> tuple[int, int]:
    """Return (cache_read, cache_write) the fake should report.

    Deterministic stand-in for Bedrock's behavior: when a cachePoint is present,
    attribute the input tokens to a cache read so tests can assert reads > 0.
    """
    if _has_cache_point(body):
        return in_tokens, 0
    return 0, 0


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/model/{model_id:path}/converse")
async def converse(model_id: str, request: Request) -> dict:
    body = await request.json()
    reply = _reply_text(body)
    in_tokens, out_tokens = _token_counts(body, reply)
    cache_read, cache_write = _cache_counts(body, in_tokens)
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": reply}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": in_tokens,
            "outputTokens": out_tokens,
            "cacheReadInputTokens": cache_read,
            "cacheWriteInputTokens": cache_write,
        },
    }


@app.post("/model/{model_id:path}/converse-stream")
async def converse_stream(model_id: str, request: Request) -> Response:
    body = await request.json()
    reply = _reply_text(body)
    in_tokens, out_tokens = _token_counts(body, reply)
    payload = encode_converse_stream(
        reply, input_tokens=in_tokens, output_tokens=out_tokens
    )
    return Response(
        content=payload,
        media_type="application/vnd.amazon.eventstream",
    )
