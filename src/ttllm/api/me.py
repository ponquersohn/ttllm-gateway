"""Self-service /me endpoints available to any authenticated user."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from ttllm.api.deps import AuthContext, DB, get_authenticated
from ttllm.schemas.admin import WhoamiResponse
from ttllm.schemas.auth import TokenCreatedResponse, TokenResponse
from ttllm.schemas.me import AvailableModelResponse, SelfTokenCreate
from ttllm.services import auth_service, group_service, model_service

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
    ctx: AuthContext = Depends(get_authenticated),
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
    ctx: AuthContext = Depends(get_authenticated),
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
    ctx: AuthContext = Depends(get_authenticated),
):
    """Revoke one of the authenticated user's own tokens."""
    token = await auth_service.get_token(db, token_id)
    if not token or token.user_id != ctx.user.id:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})
    revoked = await auth_service.revoke_token(db, token_id)
    if not revoked:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})
