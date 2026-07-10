"""POST /anthropic/v1/messages - Anthropic-compatible messages endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Annotated

from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.api.deps import AuthContext, DB, _authenticate, get_db, require_permission
from ttllm.config import settings
from ttllm.core import gateway
from ttllm.core.gateway import ServerToolError
from ttllm.core.permissions import Permissions
from ttllm.core.rules import ActionType
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.services import audit_service, model_service, rules_service, secret_service

logger = logging.getLogger(__name__)

# Map Bedrock/AWS error codes to Anthropic-compatible error types and HTTP status codes.
_BEDROCK_ERROR_MAP: dict[str, tuple[int, str]] = {
    "ThrottlingException": (529, "overloaded_error"),
    "ModelTimeoutException": (529, "overloaded_error"),
    "ModelNotReadyException": (529, "overloaded_error"),
    "ServiceUnavailableException": (529, "overloaded_error"),
    "ServiceQuotaExceededException": (529, "overloaded_error"),
    "AccessDeniedException": (403, "permission_error"),
    "ResourceNotFoundException": (404, "not_found_error"),
    "ValidationException": (400, "invalid_request_error"),
    "ModelErrorException": (500, "api_error"),
    "ModelStreamErrorException": (500, "api_error"),
    "InternalServerException": (500, "api_error"),
}


def _classify_provider_error(exc: Exception) -> tuple[int, str, str]:
    """Return (http_status, anthropic_error_type, message) for a provider exception."""
    if isinstance(exc, ServerToolError):
        return (501, "not_implemented_error", str(exc))

    if isinstance(exc, ReadTimeoutError):
        return (529, "overloaded_error", "Model request timed out — try again or use streaming")

    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        message = exc.response.get("Error", {}).get("Message", str(exc))
        if code in _BEDROCK_ERROR_MAP:
            status, error_type = _BEDROCK_ERROR_MAP[code]
            return (status, error_type, message)

    return (500, "api_error", "An internal error occurred")

router = APIRouter()


class _ModelProxy:
    """Lightweight proxy that overrides config_json with resolved secrets."""

    def __init__(self, model, resolved_config: dict):
        self._model = model
        self._resolved_config = resolved_config

    @property
    def config_json(self):
        return self._resolved_config

    def __getattr__(self, name):
        return getattr(self._model, name)


async def get_anthropic_authenticated(
    x_api_key: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    """Authenticate via x-api-key header (Anthropic-compatible API)."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Missing x-api-key header"},
        )
    return await _authenticate(x_api_key, db)


AnthropicUser = Annotated[AuthContext, Depends(
    require_permission(Permissions.LLM_INVOKE, auth_dep=get_anthropic_authenticated)
)]


@router.post("/anthropic/v1/messages")
async def create_message(
    body: MessagesRequest,
    request: Request,
    db: DB,
    ctx: AnthropicUser,
):
    """Create a message using the Anthropic Messages API format."""
    request_id = uuid.uuid4()

    # Evaluate rules engine
    active_rules = await rules_service.get_active_rules(db)
    if active_rules:
        rule_ctx = rules_service.build_request_context(
            request=body,
            headers={k.lower(): v for k, v in request.headers.items()},
            user_id=str(ctx.user.id),
        )
        # Precompute moving-window quota aggregates into rule_ctx.metadata so the
        # pure core engine can evaluate quota conditions and render block messages.
        await rules_service.populate_quota_metadata(db, active_rules, rule_ctx)
        outcome = rules_service.evaluate_rules(active_rules, rule_ctx)
        if outcome:
            if outcome.action_type == ActionType.BLOCK:
                headers = (
                    {"Retry-After": str(outcome.retry_after_seconds)}
                    if outcome.retry_after_seconds
                    else None
                )
                raise HTTPException(
                    status_code=outcome.block_status,
                    detail={"type": "policy_error", "message": outcome.block_message},
                    headers=headers,
                )
            elif outcome.action_type == ActionType.REROUTE:
                body = body.model_copy(update={"model": outcome.rerouted_model})
            elif outcome.action_type == ActionType.REWRITE:
                body = rules_service.apply_rewrite_to_request(
                    body, outcome.rewrite_pattern, outcome.rewrite_replacement
                )

    # Check model access
    llm_model = await model_service.get_model_for_user(db, ctx.user.id, body.model)
    if not llm_model:
        raise HTTPException(
            status_code=403,
            detail={
                "type": "permission_error",
                "message": f"Model '{body.model}' is not available for your account",
            },
        )

    resolved_config = await secret_service.resolve_model_config(db, llm_model.config_json or {})
    resolved_model = _ModelProxy(llm_model, resolved_config)

    metadata = {
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }

    if body.stream:
        return await _handle_streaming(body, resolved_model, ctx.user, db, request_id, metadata)
    else:
        return await _handle_invoke(body, resolved_model, ctx.user, db, request_id, metadata)


async def _finalize(
    state,
    body,
    llm_model,
    user,
    db,
    request_id,
    metadata,
    *,
    status_code: int = 200,
    error_message: str | None = None,
):
    """Write the audit row from a completed provider state.

    Shared by both the streaming and non-streaming paths — the only difference is when the
    state has been populated. The state is opaque here: we read its getters and token fields
    and never reach inside it. ``get_metadata()`` includes the provider-computed latency.
    """
    provider_metadata = state.get_metadata()
    await audit_service.log_request(
        db,
        user_id=user.id,
        model_id=llm_model.id,
        request_id=request_id,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        total_cost=str(state.get_cost()),
        latency_ms=provider_metadata.get("latency_ms", 0),
        status_code=status_code,
        error_message=error_message,
        metadata_json=metadata,
        provider_metadata=provider_metadata,
        request_body=body.model_dump() if settings.engine.log_request_bodies else None,
        response_body=state.get_response().model_dump() if settings.engine.log_request_bodies else None,
    )


async def _log_error(exc, llm_model, user, db, request_id, metadata):
    status, error_type, message = _classify_provider_error(exc)
    logger.exception("Request %s failed (type=%s)", request_id, error_type)
    await audit_service.log_request(
        db,
        user_id=user.id,
        model_id=llm_model.id,
        request_id=request_id,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        status_code=status,
        error_message=str(exc),
        metadata_json=metadata,
    )
    return status, error_type, message


async def _handle_invoke(body, llm_model, user, db, request_id, metadata):
    try:
        state = await gateway.invoke(body, llm_model, request_id)
        response = state.get_response()
        await _finalize(state, body, llm_model, user, db, request_id, metadata)
        return JSONResponse(content=response.model_dump())

    except Exception as exc:
        status, error_type, message = await _log_error(
            exc, llm_model, user, db, request_id, metadata
        )
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )


async def _handle_streaming(body, llm_model, user, db, request_id, metadata):
    try:
        state, sse_stream = gateway.stream(body, llm_model, request_id)
        client_disconnected = asyncio.Event()
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)

        async def queue_event(event: str) -> None:
            if client_disconnected.is_set():
                return

            put_task = asyncio.create_task(queue.put(event))
            disconnect_task = asyncio.create_task(client_disconnected.wait())
            done, pending = await asyncio.wait(
                {put_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

            if disconnect_task in done and not put_task.done():
                put_task.cancel()
                return

            await put_task

        async def producer() -> None:
            producer_exc: Exception | None = None
            try:
                try:
                    async for event in sse_stream:
                        await queue_event(event)
                except Exception as exc:
                    producer_exc = exc
                    logger.exception(
                        "Streaming producer failed for request %s", request_id
                    )
                    # bedrock's own error paths set state.error before yielding a frame,
                    # so only synthesize a frame here when nothing upstream did.
                    if getattr(state, "error", None) is None:
                        frame_data = {
                            "type": "error",
                            "error": {"type": "api_error", "message": str(exc)},
                        }
                        await queue_event(
                            f"event: error\ndata: {json.dumps(frame_data)}\n\n"
                        )

                stream_exc = getattr(state, "error", None) or producer_exc
                if stream_exc is not None:
                    status, _error_type, _message = _classify_provider_error(stream_exc)
                    finalize_status = status
                    finalize_error = str(stream_exc)
                elif client_disconnected.is_set():
                    finalize_status = 499
                    finalize_error = "Client disconnected during streaming response"
                else:
                    finalize_status = 200
                    finalize_error = None

                await _finalize(
                    state,
                    body,
                    llm_model,
                    user,
                    db,
                    request_id,
                    metadata,
                    status_code=finalize_status,
                    error_message=finalize_error,
                )
            finally:
                if not client_disconnected.is_set():
                    await queue.put(None)

        async def event_generator():
            producer_task = asyncio.create_task(producer())
            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event
                await producer_task
            except asyncio.CancelledError:
                client_disconnected.set()
                await asyncio.shield(producer_task)
                raise
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                    await asyncio.gather(producer_task, return_exceptions=True)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-Id": str(request_id),
            },
        )
    except Exception as exc:
        status, error_type, message = await _log_error(
            exc, llm_model, user, db, request_id, metadata
        )
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )
