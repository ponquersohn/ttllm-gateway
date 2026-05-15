"""POST /anthropic/v1/messages - Anthropic-compatible messages endpoint."""

from __future__ import annotations

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
from ttllm.core.permissions import Permissions
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.services import audit_service, model_service, secret_service

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
        status, error_type, message = _classify_provider_error(exc)
        logger.exception("Invoke failed for request %s (type=%s)", request_id, error_type)
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
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )


async def _handle_streaming(body, llm_model, user, db, request_id, metadata):
    try:
        sse_stream, collector = await gateway.stream(body, llm_model, request_id)

        async def event_generator():
            try:
                async for event in sse_stream:
                    yield event
            finally:
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
        status, error_type, message = _classify_provider_error(exc)
        logger.exception("Stream setup failed for request %s (type=%s)", request_id, error_type)
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
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )
