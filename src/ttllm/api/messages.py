"""POST /anthropic/v1/messages - Anthropic-compatible messages endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any

from botocore.exceptions import ClientError, ReadTimeoutError
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.api.deps import AuthContext, DB, _authenticate, get_db, require_permission
from ttllm.config import settings
from ttllm.core import gateway
from ttllm.core.adapters import anthropic as anthropic_adapter
from ttllm.core.adapters.anthropic_stream import encode_anthropic_sse
from ttllm.core.errors import ServerToolError, UnsupportedContentError
from ttllm.core.permissions import Permissions
from ttllm.core.quota import (
    ActualUsage,
    QuotaDenial,
    QuotaSubject,
    decide_denial,
    denial_error_type,
    denial_message,
    denial_status,
    format_amount,
)
from ttllm.core.rules import ActionType, collect_quota_ids
from ttllm.db import async_session_factory
from ttllm.schemas.anthropic import MessagesRequest, MessagesResponse
from ttllm.services import audit_service, model_service, quota_service, rules_service, secret_service, user_service
from ttllm.services.quota_service import NO_LEASE, QuotaContentionError, QuotaExceededError, QuotaLease

logger = logging.getLogger(__name__)

# How long a full stream queue may stay full before the client counts as gone. Starlette's
# disconnect handling doesn't always reach the generator, and a body may never start at all.
_STREAM_STALL_TIMEOUT_SECONDS = 120

# Stream drain tasks run independently of the response, so the client going away can't cancel
# them before they write the audit row. The event loop only weakly references tasks.
_stream_tasks: set[asyncio.Task] = set()

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

    if isinstance(exc, UnsupportedContentError):
        return (400, "invalid_request_error", str(exc))

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


@dataclass(frozen=True)
class _ModelTarget:
    """A model resolved for a user, with everything built from it."""

    llm_model: Any  # the raw ORM row, safe to snapshot
    resolved: _ModelProxy  # carries decrypted secrets; only for the provider
    audit_ctx: dict[str, Any]


async def _resolve_target(db: AsyncSession, user: Any, model_name: str) -> _ModelTarget | None:
    llm_model = await model_service.get_model_for_user(db, user.id, model_name)
    if not llm_model:
        return None

    resolved_config = await secret_service.resolve_model_config(db, llm_model.config_json or {})
    # Snapshotted from the raw `llm_model`/`user` -- *before* wrapping in _ModelProxy,
    # whose config_json property returns resolved (decrypted) secrets that must never land
    # in an audit row. Threaded through as one bundle rather than growing every handler's
    # signature by four params.
    audit_ctx = {
        "model_name": llm_model.name,
        "model_snapshot": model_service.build_model_snapshot(llm_model),
        "user_email": user.email,
        "user_snapshot": user_service.build_user_snapshot(user),
    }
    return _ModelTarget(llm_model, _ModelProxy(llm_model, resolved_config), audit_ctx)


async def _audit_quota_block(
    denial: QuotaDenial,
    message: str,
    status: int,
    target: _ModelTarget,
    user,
    request_id,
    metadata,
) -> None:
    async with async_session_factory() as session:
        await audit_service.log_request(
            session,
            user_id=user.id,
            model_id=target.llm_model.id,
            request_id=request_id,
            input_tokens=0,
            output_tokens=0,
            **target.audit_ctx,
            latency_ms=0,
            status_code=status,
            error_message=message,
            metadata_json={
                **metadata,
                "quota": {
                    "id": denial.quota_id,
                    "name": denial.name,
                    "measure": denial.measure,
                    "limit": format_amount(denial.limit),
                    "used": format_amount(denial.used),
                    "in_flight": format_amount(denial.in_flight),
                },
            },
        )


@dataclass
class _Routing:
    """How a request got from the model the client asked for to the one serving it.

    Recorded in the audit row's ``metadata_json["routing"]`` (only when something rerouted),
    so a single row per ``request_id`` explains the whole decision.
    """

    requested_model: str
    hops: list[dict[str, str]] = field(default_factory=list)

    def add(self, by: str, name: str, from_model: str, to_model: str) -> None:
        self.hops.append({"by": by, "name": name, "from": from_model, "to": to_model})

    def as_metadata(self) -> dict[str, Any] | None:
        if not self.hops:
            return None
        return {"requested_model": self.requested_model, "hops": list(self.hops)}


def _apply_terminal_rules(active_rules, build_context, body: MessagesRequest, routing: _Routing) -> MessagesRequest:
    """First match by weight decides block/reroute/rewrite/allow."""
    if not active_rules:
        return body
    outcome = rules_service.evaluate_rules(active_rules, build_context(body))
    if not outcome:
        return body
    if outcome.action_type == ActionType.BLOCK:
        raise HTTPException(
            status_code=outcome.block_status,
            detail={"type": "policy_error", "message": outcome.block_message},
        )
    if outcome.action_type == ActionType.REROUTE:
        routing.add("rule", outcome.rule_name, body.model, outcome.rerouted_model)
        return body.model_copy(update={"model": outcome.rerouted_model})
    if outcome.action_type == ActionType.REWRITE:
        return rules_service.apply_rewrite_to_request(body, outcome.rewrite_pattern, outcome.rewrite_replacement)
    return body


async def _route_request(
    db: AsyncSession,
    body: MessagesRequest,
    ctx: AuthContext,
    active_rules,
    build_context,
    request_id: uuid.UUID,
    metadata: dict[str, Any],
) -> tuple[MessagesRequest, _ModelTarget, QuotaLease]:
    """Run the whole routing pipeline: terminal rules, model access, quotas.

    A quota reroute restarts the pipeline from the client's original request with only the
    model swapped, so the fallback model gets exactly the treatment a request sent to it
    directly would: its own terminal rules, access check and quotas. Nothing is reserved
    against the original model, because a denial rolls back every reservation. At most one
    quota reroute is followed, which also bounds the loop (a rule rerouting back to an
    exhausted model just ends in a block).

    Always ends with the request session's transaction committed, so its connection isn't
    held during the provider call. Sets ``metadata["routing"]`` when anything rerouted.
    """
    routing = _Routing(body.model)
    original = body
    quota_rerouted = False
    while True:
        current = _apply_terminal_rules(active_rules, build_context, body, routing)
        if routing.hops:
            metadata["routing"] = routing.as_metadata()

        target = await _resolve_target(db, ctx.user, current.model)
        if not target:
            raise HTTPException(
                status_code=403,
                detail={
                    "type": "permission_error",
                    "message": f"Model '{current.model}' is not available for your account",
                },
            )

        # Quota rules see the request as it will actually be sent, so no terminal rule can skip them.
        quota_ids = (
            collect_quota_ids(active_rules, build_context(current, target.llm_model.name)) if active_rules else []
        )
        await db.commit()
        if not quota_ids:
            return current, target, NO_LEASE

        subject = QuotaSubject(
            user_id=str(ctx.user.id),
            user_email=ctx.user.email,
            model_id=str(target.llm_model.id),
            model_name=target.llm_model.name,
            token_id=str(ctx.jti),
        )
        try:
            lease = await quota_service.reserve(
                quota_ids, subject, quota_service.build_estimate(current, target.llm_model), request_id
            )
            return current, target, lease
        except QuotaContentionError:
            logger.warning("Quota lock contention for request %s", request_id)
            raise HTTPException(
                status_code=529,
                detail={"type": "overloaded_error", "message": "Quota system is busy, please retry"},
            )
        except QuotaExceededError as exc:
            decision = decide_denial(exc.denials)

        if decision.reroute_target and not quota_rerouted:
            quota_rerouted = True
            # Cheap access check only; the next pass resolves the target properly.
            available = await model_service.get_model_for_user(db, ctx.user.id, decision.reroute_target)
            await db.commit()
            if available:
                logger.info(
                    "Quota '%s' rerouted request %s from '%s' to '%s'",
                    decision.primary.name, request_id, target.llm_model.name, decision.reroute_target,
                )
                routing.add("quota", decision.primary.name, target.llm_model.name, decision.reroute_target)
                body = original.model_copy(update={"model": decision.reroute_target})
                continue
            logger.warning(
                "Quota '%s' reroute target '%s' isn't available to user %s",
                decision.primary.name, decision.reroute_target, ctx.user.id,
            )

        raise await _quota_block(decision, target, ctx.user, request_id, metadata)


async def _quota_block(decision, target: _ModelTarget, user, request_id, metadata) -> HTTPException:
    """Audit a quota denial and build the error response for it."""
    primary = decision.primary
    message = denial_message(primary)
    status = denial_status(primary)
    await _audit_quota_block(primary, message, status, target, user, request_id, metadata)
    headers = {
        "x-ttllm-quota-name": primary.name,
        "x-ttllm-quota-limit": format_amount(primary.limit),
        "x-ttllm-quota-remaining": "0",
    }
    if decision.retry_after is not None:
        headers["Retry-After"] = str(decision.retry_after)
        headers["x-ttllm-quota-reset"] = str(decision.retry_after)
    return HTTPException(
        status_code=status,
        detail={"type": denial_error_type(primary), "message": message},
        headers=headers,
    )


@router.post("/anthropic/v1/messages")
async def create_message(
    body: MessagesRequest,
    request: Request,
    db: DB,
    ctx: AnthropicUser,
):
    """Create a message using the Anthropic Messages API format."""
    request_id = uuid.uuid4()
    try:
        return await _create_message(body, request, db, ctx, request_id)
    except HTTPException as exc:
        # Every response carries the id of its audit row, errors included.
        exc.headers = {**(exc.headers or {}), "X-Request-Id": str(request_id)}
        raise


async def _create_message(body: MessagesRequest, request: Request, db: AsyncSession, ctx: AuthContext, request_id):
    request_headers = {k.lower(): v for k, v in request.headers.items()}

    active_rules = await rules_service.get_active_rules(db)
    groups = await rules_service.load_subject_groups(db, active_rules, ctx.user.id)

    def build_context(current: MessagesRequest, resolved_model: str | None = None):
        return rules_service.build_request_context(
            request=current,
            headers=request_headers,
            user_id=str(ctx.user.id),
            user_email=ctx.user.email,
            token_id=str(ctx.jti),
            groups=groups,
            resolved_model=resolved_model,
        )

    metadata: dict[str, Any] = {
        "client_ip": request.client.host if request.client else None,
        "user_agent": request.headers.get("user-agent"),
    }

    body, target, lease = await _route_request(db, body, ctx, active_rules, build_context, request_id, metadata)

    if body.stream:
        return await _handle_streaming(body, target.resolved, ctx.user, request_id, metadata, target.audit_ctx, lease)
    else:
        return await _handle_invoke(body, target.resolved, ctx.user, request_id, metadata, target.audit_ctx, lease)


async def _write_audit_and_settle(audit_fields: dict[str, Any], lease: QuotaLease, usage_state) -> None:
    """Write the audit row and settle the lease atomically, in a fresh session.

    A fresh session because the request's may already be closed for a stream, and because a
    failed settle leaves a session unusable. Audit is the record of what happened, so if
    settling fails it's still written, flagged, and the reservation is left to expire.
    """
    if not lease.reservations:
        async with async_session_factory() as session:
            await audit_service.log_request(session, **audit_fields)
        return

    usage = ActualUsage(
        status_code=audit_fields["status_code"],
        tokens=usage_state.get_total_tokens() if usage_state else 0,
        cost=usage_state.get_cost() if usage_state else Decimal(0),
    )
    try:
        async with async_session_factory() as session:
            await audit_service.log_request(session, **audit_fields, commit=False)
            await quota_service.settle(session, lease, usage)
            await session.commit()
        return
    except Exception:
        logger.exception("Quota settlement failed for request %s", audit_fields["request_id"])

    flagged = {
        **audit_fields,
        "metadata_json": {**(audit_fields.get("metadata_json") or {}), "quota_settle_failed": True},
    }
    async with async_session_factory() as session:
        await audit_service.log_request(session, **flagged)


async def _finalize(
    state,
    body,
    llm_model,
    user,
    request_id,
    metadata,
    audit_ctx,
    lease: QuotaLease = NO_LEASE,
    *,
    status_code: int = 200,
    error_message: str | None = None,
    response: MessagesResponse | None = None,
):
    """Write the audit row and settle quota from a completed provider state.

    Shared by both the streaming and non-streaming paths — the only difference is when the
    state has been populated. The state is opaque here: we read its getters and token fields
    and never reach inside it. ``get_metadata()`` includes the provider-computed latency.
    ``response`` is the already-converted Anthropic wire response (``state.get_response()``
    is the internal-model result, not the wire shape, so the caller converts it once via
    ``anthropic_adapter.result_to_response`` and passes it in here for the audit body).
    """
    provider_metadata = state.get_metadata()
    await _write_audit_and_settle(
        dict(
            user_id=user.id,
            model_id=llm_model.id,
            request_id=request_id,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            **audit_ctx,
            total_cost=str(state.get_cost()),
            latency_ms=provider_metadata.get("latency_ms", 0),
            status_code=status_code,
            error_message=error_message,
            metadata_json=metadata,
            provider_metadata=provider_metadata,
            request_body=body.model_dump() if settings.engine.log_request_bodies else None,
            response_body=response.model_dump() if response and settings.engine.log_request_bodies else None,
        ),
        lease,
        state,
    )


async def _log_error(exc, llm_model, user, request_id, metadata, audit_ctx, lease: QuotaLease = NO_LEASE, state=None):
    """Record a failed request. ``state`` is set when the provider call itself succeeded and
    something failed afterwards: that usage was paid for, so it's recorded and charged."""
    status, error_type, message = _classify_provider_error(exc)
    logger.exception("Request %s failed (type=%s)", request_id, error_type)
    fields: dict[str, Any] = {}
    if state is not None:
        provider_metadata = state.get_metadata()
        fields = {
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "total_cost": str(state.get_cost()),
            "latency_ms": provider_metadata.get("latency_ms", 0),
            "provider_metadata": provider_metadata,
        }
    await _write_audit_and_settle(
        {
            "user_id": user.id,
            "model_id": llm_model.id,
            "request_id": request_id,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            **audit_ctx,
            **fields,
            "status_code": status,
            "error_message": str(exc),
            "metadata_json": metadata,
        },
        lease,
        state,
    )
    return status, error_type, message


async def _handle_invoke(body, llm_model, user, request_id, metadata, audit_ctx, lease: QuotaLease = NO_LEASE):
    state = None
    try:
        internal_request = anthropic_adapter.request_to_internal(body, llm_model)
        state = await gateway.invoke(internal_request, llm_model, request_id)
        response = anthropic_adapter.result_to_response(state.get_response(), llm_model.name, request_id)
    except Exception as exc:
        status, error_type, message = await _log_error(
            exc, llm_model, user, request_id, metadata, audit_ctx, lease, state
        )
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )

    # Outside the try above: a failure from here on must not be recorded a second time.
    try:
        await _finalize(state, body, llm_model, user, request_id, metadata, audit_ctx, lease, response=response)
    except Exception:
        logger.exception("Failed to record completed request %s", request_id)
        raise HTTPException(
            status_code=500,
            detail={"type": "api_error", "message": "An internal error occurred"},
        )
    # safeguard_results only appears when safeguards actually ran, as on Anthropic's API.
    exclude = {"safeguard_results"} if response.safeguard_results is None else None
    return JSONResponse(
        content=response.model_dump(exclude=exclude),
        headers={"X-Request-Id": str(request_id), **lease.headers},
    )


async def _handle_streaming(body, llm_model, user, request_id, metadata, audit_ctx, lease: QuotaLease = NO_LEASE):
    try:
        internal_request = anthropic_adapter.request_to_internal(body, llm_model)
        state, chunk_stream = gateway.stream(internal_request, llm_model, request_id)
    except Exception as exc:
        status, error_type, message = await _log_error(
            exc, llm_model, user, request_id, metadata, audit_ctx, lease
        )
        raise HTTPException(
            status_code=status,
            detail={"type": error_type, "message": message},
        )

    sse_stream = encode_anthropic_sse(chunk_stream, llm_model.name, request_id)
    client_disconnected = asyncio.Event()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)

    def mark_disconnected() -> None:
        """Stop forwarding. Unsticks a still-attached consumer, which sees the end of stream."""
        client_disconnected.set()
        while not queue.empty():
            queue.get_nowait()
        queue.put_nowait(None)

    async def queue_event(event: str | None) -> None:
        if client_disconnected.is_set():
            return
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        put_task = asyncio.create_task(queue.put(event))
        disconnect_task = asyncio.create_task(client_disconnected.wait())
        done, pending = await asyncio.wait(
            {put_task, disconnect_task},
            timeout=_STREAM_STALL_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if not done:
            logger.warning("Client stopped reading stream for request %s", request_id)
            mark_disconnected()

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

            response = None
            if finalize_status == 200:
                try:
                    response = anthropic_adapter.result_to_response(state.get_response(), llm_model.name, request_id)
                except Exception as exc:
                    logger.exception("Failed to assemble response for request %s", request_id)
                    finalize_status, _error_type, _message = _classify_provider_error(exc)
                    finalize_error = str(exc)
            await _finalize(
                state,
                body,
                llm_model,
                user,
                request_id,
                metadata,
                audit_ctx,
                lease,
                status_code=finalize_status,
                error_message=finalize_error,
                response=response,
            )
        except Exception:
            logger.exception("Failed to record stream for request %s", request_id)
        finally:
            await queue_event(None)

    # Started here, not when the response body starts, and not owned by the response: on a
    # disconnect Starlette cancels its scope (and anyio re-delivers the cancellation past
    # asyncio.shield), which would otherwise kill this before it writes the audit row.
    producer_task = asyncio.create_task(producer())
    _stream_tasks.add(producer_task)
    producer_task.add_done_callback(_stream_tasks.discard)

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            # Normal completion must still wait: Mangum stops the event loop once the
            # response ends. Shielded so a late cancel can't kill the task from here.
            await asyncio.shield(producer_task)
        finally:
            if not producer_task.done():
                mark_disconnected()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": str(request_id),
            **lease.headers,
        },
    )
