"""Self-service /me endpoints available to any authenticated user."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ttllm.api.deps import AuthContext, DB, get_authenticated, get_session_authenticated
from ttllm.schemas.admin import CostBreakdownItem, QuotaBucketResponse, UsageSummaryResponse, WhoamiResponse
from ttllm.schemas.auth import TokenCreatedResponse, TokenResponse
from ttllm.schemas.me import (
    AvailableModelResponse,
    MyQuotaOverride,
    QuotaOverrideCode,
    QuotaOverridePreview,
    SelfTokenCreate,
)
from ttllm.services import (
    admin_audit_service,
    auth_service,
    group_service,
    model_service,
    quota_override_service,
    quota_service,
    usage_service,
)
from ttllm.services.quota_override_service import INVALID_CODE_MESSAGE, InvalidOverrideCodeError

router = APIRouter(tags=["me"])


async def _build_whoami(db: DB, ctx: AuthContext) -> WhoamiResponse:
    """Shared whoami logic used by both /me and /admin/me."""
    all_perms = await auth_service.resolve_user_permissions(db, ctx.user.id)
    groups = await group_service.list_user_groups(db, ctx.user.id)
    return WhoamiResponse(
        id=ctx.user.id,
        name=ctx.user.name,
        email=ctx.user.email,
        groups=[g.name for g in groups],
        effective_permissions=sorted(ctx.permissions),
        available_permissions=sorted(all_perms),
    )


@router.get("/me", response_model=WhoamiResponse)
async def whoami(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Return current user info, effective token permissions, and all available permissions."""
    return await _build_whoami(db, ctx)


@router.get("/me/models", response_model=list[AvailableModelResponse])
async def list_my_models(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """List models available to the authenticated user (direct + group assignments)."""
    models = await model_service.list_user_models(db, ctx.user.id)
    return [AvailableModelResponse.model_validate(m) for m in models]


@router.get("/me/tokens", response_model=list[TokenResponse])
async def list_my_tokens(
    db: DB,
    ctx: AuthContext = Depends(get_session_authenticated),
):
    """List the authenticated user's own tokens."""
    tokens = await auth_service.list_tokens(db, user_id=ctx.user.id)
    result = []
    for t in tokens:
        resp = TokenResponse.model_validate(t)
        resp.user_email = ctx.user.email
        result.append(resp)
    return result


@router.post("/me/tokens", response_model=TokenCreatedResponse, status_code=201)
async def create_my_token(
    body: SelfTokenCreate,
    db: DB,
    ctx: AuthContext = Depends(get_session_authenticated),
):
    """Create a token for the authenticated user, scoped to their own permissions."""
    try:
        return await auth_service.create_token(
            db,
            ctx.user.id,
            label=body.label,
            ttl_days=body.ttl_days,
            permissions=body.permissions,
        )
    except ValueError as e:
        raise HTTPException(400, detail={"type": "invalid_request", "message": str(e)})


@router.delete("/me/tokens/{token_id}", status_code=204)
async def revoke_my_token(
    token_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(get_session_authenticated),
):
    """Revoke one of the authenticated user's own tokens."""
    token = await auth_service.get_token(db, token_id)
    if not token or token.user_id != ctx.user.id:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})
    revoked = await auth_service.revoke_token(db, token_id)
    if not revoked:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})


@router.get("/me/usage", response_model=UsageSummaryResponse)
async def get_my_usage(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
    since: datetime | None = None,
    until: datetime | None = None,
):
    """Return usage summary for the authenticated user."""
    summary = await usage_service.get_usage_summary(
        db, user_id=ctx.user.id, since=since, until=until
    )
    return UsageSummaryResponse(**summary)


@router.get("/me/usage/costs", response_model=list[CostBreakdownItem])
async def get_my_costs(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
    since: datetime | None = None,
    until: datetime | None = None,
):
    """Return per-model cost breakdown for the authenticated user."""
    return await usage_service.get_cost_breakdown(
        db, user_id=ctx.user.id, since=since, until=until
    )


@router.get("/me/quotas", response_model=list[QuotaBucketResponse])
async def get_my_quotas(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Current standing against the quotas that apply to you. Read-only.

    Rules matching on request content can't be evaluated here, so those are best-effort.
    """
    views = await quota_service.statuses_for_user(db, ctx.user, ctx.jti)
    return [QuotaBucketResponse(**v) for v in views]


@router.get("/me/quota-overrides", response_model=list[MyQuotaOverride])
async def list_my_quota_overrides(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Your active quota overrides. While one is active, no quota applies to you."""
    rows = await quota_override_service.list_active_for_user(db, ctx.user.id)
    return [MyQuotaOverride.model_validate(r) for r in rows]


@router.post("/me/quota-overrides/lookup", response_model=QuotaOverridePreview)
async def lookup_quota_override(
    body: QuotaOverrideCode,
    db: DB,
    ctx: AuthContext = Depends(get_session_authenticated),
):
    """Preview what an override code grants, without redeeming it."""
    row = await quota_override_service.lookup(db, body.code)
    if row is None:
        raise HTTPException(404, detail={"type": "not_found", "message": INVALID_CODE_MESSAGE})
    return QuotaOverridePreview(
        reason=row.reason,
        created_by_email=row.created_by_email,
        override_duration_seconds=row.override_duration_seconds,
        code_expires_at=row.code_expires_at,
    )


@router.post("/me/quota-overrides/redeem", response_model=MyQuotaOverride)
async def redeem_quota_override(
    body: QuotaOverrideCode,
    db: DB,
    ctx: AuthContext = Depends(get_session_authenticated),
):
    """Redeem a single-use override code. Quotas stop applying to you until it expires."""
    try:
        row = await quota_override_service.redeem(db, body.code, ctx.user)
    except InvalidOverrideCodeError as e:
        await db.rollback()
        raise HTTPException(404, detail={"type": "not_found", "message": str(e)})
    result = MyQuotaOverride.model_validate(row)
    await admin_audit_service.log(
        db,
        actor=ctx.user,
        actor_jti=ctx.jti,
        action="quota_override.redeem",
        resource_type="quota_override",
        resource_id=row.id,
        details={
            "reason": row.reason,
            "issued_by": row.created_by_email,
            "override_expires_at": row.override_expires_at.isoformat(),
        },
    )
    return result
