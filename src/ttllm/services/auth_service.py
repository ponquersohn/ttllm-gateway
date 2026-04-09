"""Authentication service: login, JWT management, SSO provisioning, permission resolution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.core.jwt import (
    JWTConfig,
    create_access_token,
    create_refresh_token,
    hash_refresh_token,
)
from ttllm.core.password import verify_password
from ttllm.core.permissions import Permissions, PermissionRegistry
from ttllm.models.auth import Token, RefreshToken, UserGroup, UserPermission
from ttllm.models.user import User
from ttllm.schemas.auth import TokenCreatedResponse, LoginTokenResponse
from ttllm.services import group_service

# Lazily initialized by the app startup
_permission_registry: PermissionRegistry | None = None


def set_permission_registry(registry: PermissionRegistry) -> None:
    global _permission_registry
    _permission_registry = registry


def get_permission_registry() -> PermissionRegistry:
    if _permission_registry is None:
        raise RuntimeError("Permission registry not initialized")
    return _permission_registry


def _jwt_config() -> JWTConfig:
    return JWTConfig(
        secret_key=settings.auth.jwt.secret_key,
        algorithm=settings.auth.jwt.algorithm,
    )


# --- Local authentication ---


async def authenticate_local(
    db: AsyncSession,
    email: str,
    password: str,
) -> User | None:
    """Verify email + password for an internal user. Returns User or None."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        return None
    if not user.is_active:
        return None
    if user.identity_provider is not None:
        return None  # SSO user cannot login with password
    if not user.password_hash:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


# --- Permission resolution ---


async def resolve_user_permissions(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> set[str]:
    """Resolve all permissions for a user via group memberships + direct assignments."""
    # Permissions from groups
    groups = await group_service.list_user_groups(db, user_id)
    perms: set[str] = set()
    for g in groups:
        for gp in g.permissions_rel:
            perms.add(gp.permission)

    # Direct user permissions
    result = await db.execute(
        select(UserPermission.permission).where(UserPermission.user_id == user_id)
    )
    for row in result.all():
        perms.add(row[0])

    return perms


# --- Management tokens ---


async def create_management_tokens(
    db: AsyncSession,
    user: User,
) -> LoginTokenResponse:
    """Create a management-scoped JWT + refresh token pair."""
    registry = get_permission_registry()
    all_perms = await resolve_user_permissions(db, user.id)
    mgmt_perms = registry.filter_by_category(all_perms, "management")

    jti = uuid.uuid4()
    ttl = timedelta(minutes=settings.auth.jwt.access_token_ttl_minutes)
    access_token = create_access_token(
        user_id=user.id,
        permissions=list(mgmt_perms),
        jti=jti,
        ttl=ttl,
        config=_jwt_config(),
    )

    raw_refresh = create_refresh_token()
    refresh_expires = datetime.now(UTC) + timedelta(days=settings.auth.jwt.refresh_token_ttl_days)
    rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_expires,
    )
    db.add(rt)
    await db.commit()

    return LoginTokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=int(ttl.total_seconds()),
    )


# --- Refresh ---


async def refresh_management_token(
    db: AsyncSession,
    raw_refresh_token: str,
) -> LoginTokenResponse | None:
    """Validate a refresh token and issue a new management JWT + refresh token pair."""
    token_hash = hash_refresh_token(raw_refresh_token)
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked_at.is_(None),
        )
    )
    rt = result.scalar_one_or_none()
    if not rt:
        return None
    if rt.expires_at < datetime.now(UTC):
        return None

    # Revoke old refresh token
    rt.revoked_at = datetime.now(UTC)

    # Load user
    user = await db.get(User, rt.user_id)
    if not user or not user.is_active:
        await db.commit()
        return None

    # Re-resolve permissions (picks up any group/permission changes)
    registry = get_permission_registry()
    all_perms = await resolve_user_permissions(db, user.id)
    mgmt_perms = registry.filter_by_category(all_perms, "management")

    jti = uuid.uuid4()
    ttl = timedelta(minutes=settings.auth.jwt.access_token_ttl_minutes)
    access_token = create_access_token(
        user_id=user.id,
        permissions=list(mgmt_perms),
        jti=jti,
        ttl=ttl,
        config=_jwt_config(),
    )

    raw_refresh = create_refresh_token()
    refresh_expires = datetime.now(UTC) + timedelta(days=settings.auth.jwt.refresh_token_ttl_days)
    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw_refresh),
        expires_at=refresh_expires,
    )
    db.add(new_rt)
    await db.commit()

    return LoginTokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=int(ttl.total_seconds()),
    )


# --- Tokens ---


async def create_token(
    db: AsyncSession,
    user_id: uuid.UUID,
    label: str | None = None,
    ttl_days: int | None = None,
    permissions: list[str] | None = None,
) -> TokenCreatedResponse:
    """Create a JWT registered in the DB with specified permissions.

    If *permissions* is None, defaults to ["llm.invoke"] (gateway scope).
    Requested permissions must be a subset of the user's resolved permissions
    and must all belong to the same category (management or gateway).
    """
    registry = get_permission_registry()
    all_perms = await resolve_user_permissions(db, user_id)

    # Default to gateway llm.invoke
    requested = permissions or [Permissions.LLM_INVOKE]

    # Validate each permission exists and user has it
    for p in requested:
        if p not in registry.permissions:
            raise ValueError(f"Unknown permission: {p}")
        if p not in all_perms:
            raise ValueError(f"User does not have permission: {p}")

    # All requested permissions must belong to the same category (management or gateway)
    categories = {registry.permissions[p].category for p in requested}
    if len(categories) > 1:
        raise ValueError(
            f"Cannot mix permission categories in a single token, got: {', '.join(sorted(categories))}"
        )

    default_ttl = settings.auth.jwt.token_ttl_days
    max_ttl = settings.auth.jwt.token_max_ttl_days
    if ttl_days is not None:
        if ttl_days < 1 or ttl_days > max_ttl:
            raise ValueError(f"ttl_days must be between 1 and {max_ttl}")
    else:
        ttl_days = default_ttl
    expires_at = datetime.now(UTC) + timedelta(days=ttl_days)

    gt = Token(
        user_id=user_id,
        label=label,
        permissions=sorted(requested),
        expires_at=expires_at,
    )
    db.add(gt)
    await db.commit()
    await db.refresh(gt)

    ttl = timedelta(days=ttl_days)
    access_token = create_access_token(
        user_id=user_id,
        permissions=sorted(requested),
        jti=gt.id,
        ttl=ttl,
        config=_jwt_config(),
    )

    return TokenCreatedResponse(
        access_token=access_token,
        permissions=sorted(requested),
        id=gt.id,
        label=gt.label,
        expires_at=gt.expires_at,
    )


async def token_exists(
    db: AsyncSession,
    jti: uuid.UUID,
) -> bool:
    """Check whether a Token DB record exists for this jti."""
    return await db.get(Token, jti) is not None


async def validate_token(
    db: AsyncSession,
    jti: uuid.UUID,
) -> bool:
    """Check that a token record is active and not expired."""
    gt = await db.get(Token, jti)
    if not gt:
        return False
    if not gt.is_active:
        return False
    if gt.expires_at and gt.expires_at < datetime.now(UTC):
        return False
    return True


async def revoke_token(
    db: AsyncSession,
    token_id: uuid.UUID,
) -> bool:
    gt = await db.get(Token, token_id)
    if not gt:
        return False
    gt.is_active = False
    await db.commit()
    return True


async def get_token(db: AsyncSession, token_id: uuid.UUID) -> Token | None:
    return await db.get(Token, token_id)


async def list_tokens(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
) -> list[Token]:
    query = select(Token).where(Token.is_active == True)  # noqa: E712
    if user_id:
        query = query.where(Token.user_id == user_id)
    query = query.order_by(Token.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all())


# --- SSO provisioning ---


async def provision_sso_user(
    db: AsyncSession,
    idp_slug: str,
    user_info: dict,
    target_groups: set[str],
) -> User:
    """Look up or create a user from SSO claims. JIT provisioning.

    On every login, syncs group memberships based on target_groups
    (derived from IdP role → group_mapping + default_groups).
    """
    external_id = user_info.get("sub", "")
    email = user_info.get("email", "")
    name = user_info.get("name", email)

    # Look up existing user
    result = await db.execute(
        select(User).where(
            User.identity_provider == idp_slug,
            User.external_id == external_id,
        )
    )
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            name=name,
            email=email,
            identity_provider=idp_slug,
            external_id=external_id,
        )
        db.add(user)
        await db.flush()

    # Sync group memberships on every login
    if target_groups:
        await _sync_user_groups(db, user, target_groups)

    await db.commit()
    await db.refresh(user)
    return user


async def _sync_user_groups(
    db: AsyncSession,
    user: User,
    target_group_names: set[str],
) -> None:
    """Ensure user belongs to all target groups. Adds missing memberships.

    Raises ValueError if any target group names don't exist in the DB.
    """
    from ttllm.models.auth import Group

    grp_result = await db.execute(
        select(Group).where(
            Group.name.in_(target_group_names),
            Group.is_active == True,  # noqa: E712
        )
    )
    found_groups = list(grp_result.scalars().all())
    found_names = {g.name for g in found_groups}
    missing = target_group_names - found_names
    if missing:
        raise ValueError(f"SSO group sync failed: groups not found in DB: {', '.join(sorted(missing))}")

    existing = await db.execute(
        select(UserGroup.group_id).where(UserGroup.user_id == user.id)
    )
    existing_group_ids = {row[0] for row in existing.all()}

    for group in found_groups:
        if group.id not in existing_group_ids:
            db.add(UserGroup(user_id=user.id, group_id=group.id))


# --- User-level permission management ---


async def assign_user_permission(
    db: AsyncSession,
    user_id: uuid.UUID,
    permission: str,
) -> UserPermission:
    up = UserPermission(user_id=user_id, permission=permission)
    db.add(up)
    await db.commit()
    await db.refresh(up)
    return up


async def unassign_user_permission(
    db: AsyncSession,
    user_id: uuid.UUID,
    permission: str,
) -> bool:
    result = await db.execute(
        select(UserPermission).where(
            UserPermission.user_id == user_id, UserPermission.permission == permission
        )
    )
    up = result.scalar_one_or_none()
    if not up:
        return False
    await db.delete(up)
    await db.commit()
    return True


async def list_user_permissions(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[str]:
    result = await db.execute(
        select(UserPermission.permission).where(UserPermission.user_id == user_id)
    )
    return [row[0] for row in result.all()]
