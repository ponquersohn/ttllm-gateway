"""Admin API endpoints for managing users, models, assignments, groups, tokens, and usage."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text

from ttllm import __version__
from ttllm.api.deps import AuthContext, DB, get_authenticated, require_permission
from ttllm.config import settings
from ttllm.core.permissions import Permissions
from ttllm.core.secrets import validate_fernet_key
from ttllm.schemas.admin import (
    AssignRequest,
    AuditLogBodyResponse,
    AuditLogResponse,
    CostBreakdownItem,
    GroupAssignRequest,
    ModelCreate,
    ModelResponse,
    ModelUpdate,
    RuleCreate,
    RuleResponse,
    RuleUpdate,
    SecretCreate,
    SecretResponse,
    SecretUpdate,
    ServerStatusResponse,
    StatusCheck,
    UsageSummaryResponse,
    UserCreate,
    UserResponse,
    UserUpdate,
    WhoamiResponse,
)
from ttllm.schemas.auth import (
    TokenCreate,
    TokenCreatedResponse,
    TokenResponse,
    GroupCreate,
    GroupMemberAssign,
    GroupPermissionAssign,
    GroupResponse,
    GroupUpdate,
    UserPermissionAssign,
)
from ttllm.schemas.common import PaginatedResponse
from ttllm.services import admin_audit_service, audit_service, auth_service, group_service, model_service, rules_service, secret_service, usage_service, user_service
from ttllm.api.me import _build_whoami

router = APIRouter(prefix="/admin", tags=["admin"])


# --- Me ---


@router.get("/me", response_model=WhoamiResponse)
async def whoami(
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Return current user info (alias for GET /me, kept for backward compatibility)."""
    return await _build_whoami(db, ctx)


# --- Status ---


@router.get("/status", response_model=ServerStatusResponse)
async def server_status(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SERVER_STATUS)),
):
    """Return server version, status, and configuration health checks."""
    checks: list[StatusCheck] = []

    # -- encryption_key check --
    enc_key = settings.secrets.encryption_key
    if not enc_key:
        checks.append(StatusCheck(name="encryption_key", status="error", message="Encryption key is not set"))
    elif not validate_fernet_key(enc_key):
        checks.append(StatusCheck(name="encryption_key", status="error", message="Encryption key is not a valid Fernet key"))
    else:
        checks.append(StatusCheck(name="encryption_key", status="ok"))

    # -- jwt_secret check --
    if settings.auth.jwt.secret_key == "CHANGE-ME-IN-PRODUCTION":
        checks.append(StatusCheck(name="jwt_secret", status="warning", message="JWT secret is using the default value"))
    else:
        checks.append(StatusCheck(name="jwt_secret", status="ok"))

    # -- database check --
    try:
        await db.execute(text("SELECT 1"))
        checks.append(StatusCheck(name="database", status="ok"))
    except Exception:
        checks.append(StatusCheck(name="database", status="error", message="Database unreachable"))

    overall = "ok" if all(c.status == "ok" for c in checks) else "degraded"
    return ServerStatusResponse(version=__version__, status=overall, checks=checks)


# --- Helpers ---


def _user_response(user, groups=None) -> UserResponse:
    group_names = [g.name for g in groups] if groups else []
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        identity_provider=user.identity_provider,
        groups=group_names,
        is_active=user.is_active,
        created_at=user.created_at,
    )


def _group_response(group) -> GroupResponse:
    return GroupResponse(
        id=group.id,
        name=group.name,
        description=group.description,
        permissions=[gp.permission for gp in group.permissions_rel],
        is_active=group.is_active,
        created_at=group.created_at,
    )


# --- Users ---


@router.get("/users", response_model=PaginatedResponse[UserResponse])
async def list_users(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_VIEW)),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    users, total = await user_service.list_users(db, offset=offset, limit=limit)
    items = []
    for u in users:
        groups = await group_service.list_user_groups(db, u.id)
        items.append(_user_response(u, groups))
    return PaginatedResponse(items=items, total=total, offset=offset, limit=limit)


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(
    body: UserCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_CREATE)),
):
    try:
        user = await user_service.create_user(
            db, name=body.name, email=body.email, password=body.password
        )
    except ValueError as e:
        raise HTTPException(400, detail={"type": "invalid_request", "message": str(e)})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="user.create", resource_type="user", resource_id=user.id,
        details={"email": body.email},
    )
    return _user_response(user)


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_VIEW)),
):
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    groups = await group_service.list_user_groups(db, user.id)
    return _user_response(user, groups)


@router.get("/users/{user_id}/models", response_model=list[ModelResponse])
async def list_user_models(
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_VIEW)),
):
    """List all models a user can access (direct assignments + group assignments)."""
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    models = await model_service.list_user_models(db, user_id)
    return [ModelResponse.model_validate(m) for m in models]


@router.patch("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: uuid.UUID,
    body: UserUpdate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_MODIFY)),
):
    user = await user_service.update_user(db, user_id, **body.model_dump(exclude_unset=True))
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="user.update", resource_type="user", resource_id=user_id,
        details={"fields": list(body.model_dump(exclude_unset=True).keys())},
    )
    groups = await group_service.list_user_groups(db, user.id)
    return _user_response(user, groups)


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_DELETE)),
):
    user = await user_service.deactivate_user(db, user_id)
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="user.delete", resource_type="user", resource_id=user_id,
    )


# --- User Permissions ---


@router.get("/users/{user_id}/permissions")
async def get_user_permissions(
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_VIEW)),
):
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    direct = await auth_service.list_user_permissions(db, user_id)
    effective = await auth_service.resolve_user_permissions(db, user_id)
    return {
        "direct_permissions": sorted(direct),
        "effective_permissions": sorted(effective),
    }


@router.post("/users/{user_id}/permissions", status_code=201)
async def assign_user_permissions(
    user_id: uuid.UUID,
    body: UserPermissionAssign,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_MODIFY)),
):
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(404, detail={"type": "not_found", "message": "User not found"})
    registry = auth_service.get_permission_registry()
    for perm in body.permissions:
        if not registry.validate_permission(perm):
            raise HTTPException(400, detail={"type": "invalid_request", "message": f"Unknown permission: {perm}"})
    results = []
    for perm in body.permissions:
        try:
            await auth_service.assign_user_permission(db, user_id, perm)
            results.append({"permission": perm, "status": "assigned"})
        except Exception:
            results.append({"permission": perm, "status": "already_assigned"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="user.assign_permission", resource_type="user", resource_id=user_id,
        details={"permissions": body.permissions},
    )
    return {"permissions": results}


@router.delete("/users/{user_id}/permissions/{permission}", status_code=204)
async def unassign_user_permission(
    user_id: uuid.UUID,
    permission: str,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USER_MODIFY)),
):
    removed = await auth_service.unassign_user_permission(db, user_id, permission)
    if not removed:
        raise HTTPException(404, detail={"type": "not_found", "message": "Permission assignment not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="user.unassign_permission", resource_type="user", resource_id=user_id,
        details={"permission": permission},
    )


# --- Models ---


@router.get("/models", response_model=PaginatedResponse[ModelResponse])
async def list_models(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_VIEW)),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    models, total = await model_service.list_models(db, offset=offset, limit=limit)
    return PaginatedResponse(
        items=[ModelResponse.model_validate(m) for m in models],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(
    model_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_VIEW)),
):
    model = await model_service.get_model(db, model_id)
    if not model:
        raise HTTPException(404, detail={"type": "not_found", "message": "Model not found"})
    return ModelResponse.model_validate(model)


@router.post("/models", response_model=ModelResponse, status_code=201)
async def create_model(
    body: ModelCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_CREATE)),
):
    model = await model_service.create_model(
        db,
        name=body.name,
        provider=body.provider,
        provider_model_id=body.provider_model_id,
        config_json=body.config_json,
        input_cost_per_1k=body.input_cost_per_1k,
        output_cost_per_1k=body.output_cost_per_1k,
        cache_read_cost_per_1k=body.cache_read_cost_per_1k,
        cache_write_cost_per_1k=body.cache_write_cost_per_1k,
        match_pattern=body.match_pattern,
    )
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.create", resource_type="model", resource_id=model.id,
        details={"name": body.name},
    )
    return ModelResponse.model_validate(model)


@router.patch("/models/{model_id}", response_model=ModelResponse)
async def update_model(
    model_id: uuid.UUID,
    body: ModelUpdate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_MODIFY)),
):
    model = await model_service.update_model(
        db, model_id, **body.model_dump(exclude_unset=True)
    )
    if not model:
        raise HTTPException(404, detail={"type": "not_found", "message": "Model not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.update", resource_type="model", resource_id=model_id,
        details={"fields": [k for k in body.model_dump(exclude_unset=True) if k != "merge_config"]},
    )
    return ModelResponse.model_validate(model)


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_DELETE)),
):
    deleted = await model_service.delete_model(db, model_id)
    if not deleted:
        raise HTTPException(404, detail={"type": "not_found", "message": "Model not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.delete", resource_type="model", resource_id=model_id,
    )


# --- Assignments ---


@router.post("/models/{model_id}/assign", status_code=201)
async def assign_model(
    model_id: uuid.UUID,
    body: AssignRequest,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_ASSIGN)),
):
    model = await model_service.get_model(db, model_id)
    if not model:
        raise HTTPException(404, detail={"type": "not_found", "message": "Model not found"})

    results = []
    for user_id in body.user_ids:
        try:
            await model_service.assign_model_to_user(db, model_id, user_id)
            results.append({"user_id": str(user_id), "status": "assigned"})
        except Exception:
            results.append({"user_id": str(user_id), "status": "already_assigned"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.assign_to_user", resource_type="model", resource_id=model_id,
        details={"user_ids": [str(uid) for uid in body.user_ids]},
    )
    return {"assignments": results}


@router.delete("/models/{model_id}/assign/{user_id}", status_code=204)
async def unassign_model(
    model_id: uuid.UUID,
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_ASSIGN)),
):
    removed = await model_service.unassign_model_from_user(db, model_id, user_id)
    if not removed:
        raise HTTPException(
            404, detail={"type": "not_found", "message": "Assignment not found"}
        )
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.unassign_from_user", resource_type="model", resource_id=model_id,
        details={"user_id": str(user_id)},
    )


@router.post("/models/{model_id}/assign-group", status_code=201)
async def assign_model_to_group(
    model_id: uuid.UUID,
    body: GroupAssignRequest,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_ASSIGN)),
):
    model = await model_service.get_model(db, model_id)
    if not model:
        raise HTTPException(404, detail={"type": "not_found", "message": "Model not found"})

    results = []
    for group_id in body.group_ids:
        try:
            await model_service.assign_model_to_group(db, model_id, group_id)
            results.append({"group_id": str(group_id), "status": "assigned"})
        except Exception:
            results.append({"group_id": str(group_id), "status": "already_assigned"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.assign_to_group", resource_type="model", resource_id=model_id,
        details={"group_ids": [str(gid) for gid in body.group_ids]},
    )
    return {"assignments": results}


@router.delete("/models/{model_id}/assign-group/{group_id}", status_code=204)
async def unassign_model_from_group(
    model_id: uuid.UUID,
    group_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.MODEL_ASSIGN)),
):
    removed = await model_service.unassign_model_from_group(db, model_id, group_id)
    if not removed:
        raise HTTPException(
            404, detail={"type": "not_found", "message": "Assignment not found"}
        )
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="model.unassign_from_group", resource_type="model", resource_id=model_id,
        details={"group_id": str(group_id)},
    )


# --- Groups ---


@router.get("/groups", response_model=PaginatedResponse[GroupResponse])
async def list_groups(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_VIEW)),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    groups, total = await group_service.list_groups(db, offset=offset, limit=limit)
    return PaginatedResponse(
        items=[_group_response(g) for g in groups],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/groups", response_model=GroupResponse, status_code=201)
async def create_group(
    body: GroupCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_CREATE)),
):
    group = await group_service.create_group(db, name=body.name, description=body.description)
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.create", resource_type="group", resource_id=group.id,
        details={"name": body.name},
    )
    return _group_response(group)


@router.get("/groups/{group_id}", response_model=GroupResponse)
async def get_group(
    group_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_VIEW)),
):
    group = await group_service.get_group(db, group_id)
    if not group:
        raise HTTPException(404, detail={"type": "not_found", "message": "Group not found"})
    return _group_response(group)


@router.patch("/groups/{group_id}", response_model=GroupResponse)
async def update_group(
    group_id: uuid.UUID,
    body: GroupUpdate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_MODIFY)),
):
    group = await group_service.update_group(db, group_id, **body.model_dump(exclude_unset=True))
    if not group:
        raise HTTPException(404, detail={"type": "not_found", "message": "Group not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.update", resource_type="group", resource_id=group_id,
        details={"fields": list(body.model_dump(exclude_unset=True).keys())},
    )
    return _group_response(group)


@router.delete("/groups/{group_id}", status_code=204)
async def delete_group(
    group_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_DELETE)),
):
    deleted = await group_service.delete_group(db, group_id)
    if not deleted:
        raise HTTPException(404, detail={"type": "not_found", "message": "Group not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.delete", resource_type="group", resource_id=group_id,
    )


@router.post("/groups/{group_id}/permissions", status_code=201)
async def assign_group_permission(
    group_id: uuid.UUID,
    body: GroupPermissionAssign,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_MODIFY)),
):
    registry = auth_service.get_permission_registry()
    if not registry.validate_permission(body.permission):
        raise HTTPException(400, detail={"type": "invalid_request", "message": f"Unknown permission: {body.permission}"})
    try:
        await group_service.assign_permission(db, group_id, body.permission)
    except Exception:
        raise HTTPException(409, detail={"type": "conflict", "message": "Permission already assigned"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.assign_permission", resource_type="group", resource_id=group_id,
        details={"permission": body.permission},
    )
    return {"status": "assigned"}


@router.delete("/groups/{group_id}/permissions/{permission}", status_code=204)
async def unassign_group_permission(
    group_id: uuid.UUID,
    permission: str,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_MODIFY)),
):
    removed = await group_service.unassign_permission(db, group_id, permission)
    if not removed:
        raise HTTPException(404, detail={"type": "not_found", "message": "Permission assignment not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.unassign_permission", resource_type="group", resource_id=group_id,
        details={"permission": permission},
    )


@router.post("/groups/{group_id}/members", status_code=201)
async def add_group_members(
    group_id: uuid.UUID,
    body: GroupMemberAssign,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_MODIFY)),
):
    results = []
    for user_id in body.user_ids:
        try:
            await group_service.add_member(db, group_id, user_id)
            results.append({"user_id": str(user_id), "status": "added"})
        except Exception:
            results.append({"user_id": str(user_id), "status": "already_member"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.add_member", resource_type="group", resource_id=group_id,
        details={"user_ids": [str(uid) for uid in body.user_ids]},
    )
    return {"members": results}


@router.delete("/groups/{group_id}/members/{user_id}", status_code=204)
async def remove_group_member(
    group_id: uuid.UUID,
    user_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.GROUP_MODIFY)),
):
    removed = await group_service.remove_member(db, group_id, user_id)
    if not removed:
        raise HTTPException(404, detail={"type": "not_found", "message": "Member not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="group.remove_member", resource_type="group", resource_id=group_id,
        details={"user_id": str(user_id)},
    )


# --- Gateway Tokens ---


@router.post("/tokens", response_model=TokenCreatedResponse, status_code=201)
async def create_token(
    body: TokenCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.TOKEN_CREATE)),
):
    target_user_id = body.user_id or ctx.user.id
    try:
        result = await auth_service.create_token(
            db, target_user_id, label=body.label, ttl_days=body.ttl_days, permissions=body.permissions,
        )
    except ValueError as e:
        raise HTTPException(400, detail={"type": "invalid_request", "message": str(e)})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="token.create", resource_type="token", resource_id=result.id,
        details={"label": body.label, "target_user_id": str(target_user_id)},
    )
    return result


@router.get("/tokens/{token_id}", response_model=TokenResponse)
async def get_token_detail(
    token_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.TOKEN_CREATE)),
):
    token = await auth_service.get_token(db, token_id)
    if not token:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})
    resp = TokenResponse.model_validate(token)
    user = await user_service.get_user(db, token.user_id)
    if user:
        resp.user_email = user.email
    return resp


@router.get("/tokens", response_model=list[TokenResponse])
async def list_tokens(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.TOKEN_CREATE)),
    user_id: uuid.UUID | None = None,
):
    tokens = await auth_service.list_tokens(db, user_id=user_id)
    user_ids = {t.user_id for t in tokens}
    users = {uid: await user_service.get_user(db, uid) for uid in user_ids}
    result = []
    for t in tokens:
        resp = TokenResponse.model_validate(t)
        u = users.get(t.user_id)
        if u:
            resp.user_email = u.email
        result.append(resp)
    return result


@router.delete("/tokens/{token_id}", status_code=204)
async def revoke_token(
    token_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.TOKEN_DELETE)),
):
    revoked = await auth_service.revoke_token(db, token_id)
    if not revoked:
        raise HTTPException(404, detail={"type": "not_found", "message": "Token not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="token.revoke", resource_type="token", resource_id=token_id,
    )


# --- Usage ---


@router.get("/usage", response_model=UsageSummaryResponse)
async def get_usage(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USAGE_VIEW)),
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
):
    summary = await usage_service.get_usage_summary(
        db, user_id=user_id, model_id=model_id, since=since, until=until
    )
    return UsageSummaryResponse(**summary)


@router.get("/usage/costs", response_model=list[CostBreakdownItem])
async def get_costs(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.USAGE_VIEW)),
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
):
    return await usage_service.get_cost_breakdown(
        db, user_id=user_id, model_id=model_id, since=since, until=until
    )


# --- Secrets ---


@router.get("/secrets", response_model=PaginatedResponse[SecretResponse])
async def list_secrets(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SECRET_VIEW)),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    secrets, total = await secret_service.list_secrets(db, offset=offset, limit=limit)
    return PaginatedResponse(
        items=[SecretResponse.model_validate(s) for s in secrets],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/secrets", response_model=SecretResponse, status_code=201)
async def create_secret(
    body: SecretCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SECRET_CREATE)),
):
    existing = await secret_service.get_secret_by_name(db, body.name)
    if existing:
        raise HTTPException(409, detail={"type": "conflict", "message": f"Secret '{body.name}' already exists"})
    secret = await secret_service.create_secret(
        db, name=body.name, plaintext_value=body.value, description=body.description,
    )
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="secret.create", resource_type="secret", resource_id=secret.id,
        details={"name": body.name},
    )
    return SecretResponse.model_validate(secret)


@router.get("/secrets/{secret_id}", response_model=SecretResponse)
async def get_secret(
    secret_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SECRET_VIEW)),
):
    secret = await secret_service.get_secret(db, secret_id)
    if not secret:
        raise HTTPException(404, detail={"type": "not_found", "message": "Secret not found"})
    return SecretResponse.model_validate(secret)


@router.patch("/secrets/{secret_id}", response_model=SecretResponse)
async def update_secret(
    secret_id: uuid.UUID,
    body: SecretUpdate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SECRET_MODIFY)),
):
    kwargs: dict = {}
    if body.value is not None:
        kwargs["plaintext_value"] = body.value
    if body.description is not None:
        kwargs["description"] = body.description
    if not kwargs:
        raise HTTPException(400, detail={"type": "invalid_request", "message": "Nothing to update"})
    secret = await secret_service.update_secret(db, secret_id, **kwargs)
    if not secret:
        raise HTTPException(404, detail={"type": "not_found", "message": "Secret not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="secret.update", resource_type="secret", resource_id=secret_id,
        details={"fields": list(kwargs.keys())},
    )
    return SecretResponse.model_validate(secret)


@router.delete("/secrets/{secret_id}", status_code=204)
async def delete_secret(
    secret_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.SECRET_DELETE)),
):
    deleted = await secret_service.delete_secret(db, secret_id)
    if not deleted:
        raise HTTPException(404, detail={"type": "not_found", "message": "Secret not found"})
    await admin_audit_service.log(
        db, actor_id=ctx.user.id, actor_jti=ctx.jti,
        action="secret.delete", resource_type="secret", resource_id=secret_id,
    )


# --- Audit Logs ---


@router.get("/audit-logs", response_model=PaginatedResponse[AuditLogResponse])
async def list_audit_logs(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.AUDIT_VIEW)),
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    logs, total = await audit_service.get_audit_logs(
        db,
        user_id=user_id,
        model_id=model_id,
        since=since,
        until=until,
        offset=offset,
        limit=limit,
    )
    return PaginatedResponse(
        items=[AuditLogResponse.model_validate(log) for log in logs],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/audit-logs/{audit_log_id}/body", response_model=AuditLogBodyResponse)
async def get_audit_log_body(
    audit_log_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.AUDIT_VIEW)),
):
    body = await audit_service.get_audit_log_body(db, audit_log_id)
    if not body:
        raise HTTPException(
            404,
            detail={"type": "not_found", "message": "Audit log body not found"},
        )
    return AuditLogBodyResponse.model_validate(body)


# --- Rules ---


@router.get("/rules", response_model=PaginatedResponse[RuleResponse])
async def list_rules(
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.RULE_VIEW)),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    rules, total = await rules_service.list_rules(db, offset=offset, limit=limit)
    return PaginatedResponse(
        items=[RuleResponse.model_validate(r) for r in rules],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/rules", response_model=RuleResponse, status_code=201)
async def create_rule(
    body: RuleCreate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.RULE_CREATE)),
):
    rule = await rules_service.create_rule(
        db,
        name=body.name,
        conditions=body.conditions,
        action=body.action,
        description=body.description,
        weight=body.weight,
        enabled=body.enabled,
    )
    await admin_audit_service.log(
        db,
        actor_id=ctx.user.id,
        actor_jti=ctx.jti,
        action="rule.create",
        resource_type="rule",
        resource_id=rule.id,
        details={"name": body.name},
    )
    return RuleResponse.model_validate(rule)


@router.get("/rules/{rule_id}", response_model=RuleResponse)
async def get_rule(
    rule_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.RULE_VIEW)),
):
    rule = await rules_service.get_rule(db, rule_id)
    if not rule:
        raise HTTPException(404, detail={"type": "not_found", "message": "Rule not found"})
    return RuleResponse.model_validate(rule)


@router.patch("/rules/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: uuid.UUID,
    body: RuleUpdate,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.RULE_MODIFY)),
):
    rule = await rules_service.update_rule(
        db, rule_id, **body.model_dump(exclude_unset=True)
    )
    if not rule:
        raise HTTPException(404, detail={"type": "not_found", "message": "Rule not found"})
    await admin_audit_service.log(
        db,
        actor_id=ctx.user.id,
        actor_jti=ctx.jti,
        action="rule.update",
        resource_type="rule",
        resource_id=rule_id,
        details={"fields": list(body.model_dump(exclude_unset=True).keys())},
    )
    return RuleResponse.model_validate(rule)


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(
    rule_id: uuid.UUID,
    db: DB,
    ctx: AuthContext = Depends(require_permission(Permissions.RULE_DELETE)),
):
    deleted = await rules_service.delete_rule(db, rule_id)
    if not deleted:
        raise HTTPException(404, detail={"type": "not_found", "message": "Rule not found"})
    await admin_audit_service.log(
        db,
        actor_id=ctx.user.id,
        actor_jti=ctx.jti,
        action="rule.delete",
        resource_type="rule",
        resource_id=rule_id,
    )
