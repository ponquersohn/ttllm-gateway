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


@app.get("/")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/model/{model_id:path}/converse")
async def converse(model_id: str, request: Request) -> dict:
    body = await request.json()
    reply = _reply_text(body)
    in_tokens, out_tokens = _token_counts(body, reply)
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": reply}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": in_tokens,
            "outputTokens": out_tokens,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
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
