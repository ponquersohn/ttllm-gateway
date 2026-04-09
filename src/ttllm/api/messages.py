"""POST /anthropic/v1/messages - Anthropic-compatible messages endpoint."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.api.deps import AuthContext, DB, _authenticate, get_db, require_permission
from ttllm.config import settings
from ttllm.core import gateway
from ttllm.core.permissions import Permissions
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.services import audit_service, model_service, secret_service

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

    # Resolve secret:// references in model config without mutating the ORM object
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


async def _handle_invoke(body, llm_model, user, db, request_id, metadata):
    try:
        result = await gateway.invoke(body, llm_model, request_id)

        # Write audit log
        await audit_service.log_request(
            db,
            user_id=user.id,
            model_id=llm_model.id,
            request_id=request_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_cost=str(result.cost),
            latency_ms=result.latency_ms,
            status_code=200,
            metadata_json=metadata,
            request_body=body.model_dump() if settings.engine.log_request_bodies else None,
            response_body=result.response.model_dump() if settings.engine.log_request_bodies else None,
        )

        return JSONResponse(content=result.response.model_dump())

    except Exception as exc:
        await audit_service.log_request(
            db,
            user_id=user.id,
            model_id=llm_model.id,
            request_id=request_id,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            status_code=500,
            error_message=str(exc),
            metadata_json=metadata,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "type": "api_error",
                "message": f"Internal error: {exc}",
            },
        )


async def _handle_streaming(body, llm_model, user, db, request_id, metadata):
    try:
        sse_stream, collector = await gateway.stream(body, llm_model, request_id)

        async def event_generator():
            async for event in sse_stream:
                yield event

            # After stream completes, write audit log
            stream_result = collector.finalize()
            await audit_service.log_request(
                db,
                user_id=user.id,
                model_id=llm_model.id,
                request_id=request_id,
                input_tokens=stream_result.input_tokens,
                output_tokens=stream_result.output_tokens,
                total_cost=str(stream_result.cost),
                latency_ms=stream_result.latency_ms,
                status_code=200,
                metadata_json=metadata,
                request_body=body.model_dump() if settings.engine.log_request_bodies else None,
            )

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
        await audit_service.log_request(
            db,
            user_id=user.id,
            model_id=llm_model.id,
            request_id=request_id,
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            status_code=500,
            error_message=str(exc),
            metadata_json=metadata,
        )
        raise HTTPException(
            status_code=500,
            detail={"type": "api_error", "message": f"Internal error: {exc}"},
        )
