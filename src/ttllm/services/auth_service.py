"""Authentication service: login, JWT management, SSO provisioning, permission resolution."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

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
    )


# --- Local authentication ---


_DUMMY_HASH = "$2b$12$LJ3m4ys3Lg2HEOFqIHwJNOd1bnMqGgYqJmGx4Q7GkWbM0FHXN0kCq"


async def authenticate_local(
    db: AsyncSession,
    email: str,
    password: str,
) -> User | None:
    """Verify email + password for an internal user. Returns User or None."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not user.is_active or user.identity_provider is not None or not user.password_hash:
        verify_password(password, _DUMMY_HASH)
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
        if ttl_days == 0:
            ttl = timedelta(seconds=10)
        elif 1 <= ttl_days <= max_ttl:
            ttl = timedelta(days=ttl_days)
        else:
            raise ValueError(f"ttl_days must be 0 (ephemeral) or between 1 and {max_ttl}")
    else:
        ttl = timedelta(days=default_ttl)
    expires_at = datetime.now(UTC) + ttl

    gt = Token(
        user_id=user_id,
        label=label,
        permissions=sorted(requested),
        expires_at=expires_at,
    )
    db.add(gt)
    await db.commit()
    await db.refresh(gt)

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
    sso_managed_groups: set[str] | None = None,
    idp_refresh_token: str | None = None,
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

    # Store encrypted IdP refresh token for background role sync
    if idp_refresh_token:
        encryption_key = settings.secrets.encryption_key
        if encryption_key:
            from ttllm.core.secrets import encrypt_value
            user.idp_refresh_token = encrypt_value(idp_refresh_token, encryption_key)
        else:
            logger.warning("Cannot store IdP refresh token: secrets.encryption_key not configured")

    user.last_role_sync_at = datetime.now(UTC)

    # Sync group memberships on every login
    await _sync_user_groups(db, user, target_groups, sso_managed_groups or target_groups)

    await db.commit()
    await db.refresh(user)
    return user


async def maybe_refresh_sso_roles(db: AsyncSession, user: User) -> None:
    """Lazy per-request check: refresh SSO user's roles from the IdP if the
    configured interval has elapsed. Uses SELECT FOR UPDATE with double-check
    to prevent concurrent races on refresh token rotation.
    """
    if not user.identity_provider or not user.idp_refresh_token:
        return

    idp_config = settings.auth.identity_providers.get(user.identity_provider)
    if not idp_config:
        return

    now = datetime.now(UTC)
    if user.last_role_sync_at and (now - user.last_role_sync_at).total_seconds() < idp_config.role_refresh_interval_seconds:
        return

    # Acquire row lock and double-check (another request may have refreshed)
    result = await db.execute(
        select(User).where(User.id == user.id).with_for_update()
    )
    locked_user = result.scalar_one()

    if locked_user.last_role_sync_at and (now - locked_user.last_role_sync_at).total_seconds() < idp_config.role_refresh_interval_seconds:
        return

    encryption_key = settings.secrets.encryption_key
    if not encryption_key:
        logger.warning("Cannot refresh SSO roles for user %s: encryption_key not configured", user.id)
        return

    from ttllm.core.secrets import decrypt_value, encrypt_value

    try:
        plain_refresh_token = decrypt_value(locked_user.idp_refresh_token, encryption_key)
    except Exception:
        logger.warning("Failed to decrypt refresh token for user %s, clearing it", user.id)
        locked_user.idp_refresh_token = None
        locked_user.last_role_sync_at = now
        await db.commit()
        return

    from ttllm.core import oidc

    try:
        endpoints = await oidc.discover(idp_config.get_discovery_url())
        token_data = await oidc.refresh_tokens(
            endpoints=endpoints,
            client_id=idp_config.client_id,
            client_secret=idp_config.client_secret,
            refresh_token=plain_refresh_token,
        )
    except Exception as exc:
        logger.warning("SSO role refresh failed for user %s: %s", user.id, exc)
        locked_user.idp_refresh_token = None
        locked_user.last_role_sync_at = now
        await db.commit()
        return

    # Store rotated refresh token
    new_refresh_token = token_data.get("refresh_token")
    if new_refresh_token:
        locked_user.idp_refresh_token = encrypt_value(new_refresh_token, encryption_key)

    # Extract roles from fresh ID token
    id_token_raw = token_data.get("id_token", "")
    target_groups = set(idp_config.default_groups)
    if id_token_raw:
        try:
            id_payload = oidc.verify_id_token(
                id_token_raw,
                endpoints=endpoints,
                client_id=idp_config.client_id,
            )
            idp_roles = oidc.extract_roles_from_id_token_payload(id_payload)
            for role in idp_roles:
                target_groups.update(idp_config.group_mapping.get(role, []))
        except ValueError as exc:
            logger.warning("ID token verification failed during role refresh for user %s: %s", user.id, exc)

    sso_managed_groups = set(idp_config.default_groups)
    for mapped in idp_config.group_mapping.values():
        sso_managed_groups.update(mapped)

    await _sync_user_groups(db, locked_user, target_groups, sso_managed_groups)
    locked_user.last_role_sync_at = now
    await db.commit()


async def _sync_user_groups(
    db: AsyncSession,
    user: User,
    target_group_names: set[str],
    sso_managed_group_names: set[str],
) -> None:
    """Sync SSO-managed group memberships: add missing, remove stale.

    Only removes memberships for groups in sso_managed_group_names that are
    not in target_group_names. Manually-assigned groups are left untouched.
    """
    from ttllm.models.auth import Group

    all_managed_names = sso_managed_group_names | target_group_names
    grp_result = await db.execute(
        select(Group).where(
            Group.name.in_(all_managed_names),
            Group.is_active == True,  # noqa: E712
        )
    )
    all_groups = list(grp_result.scalars().all())
    groups_by_name = {g.name: g for g in all_groups}

    missing = target_group_names - groups_by_name.keys()
    if missing:
        raise ValueError(f"SSO group sync failed: groups not found in DB: {', '.join(sorted(missing))}")

    existing = await db.execute(
        select(UserGroup).where(UserGroup.user_id == user.id)
    )
    existing_memberships = {ug.group_id: ug for ug in existing.scalars().all()}

    target_group_ids = {groups_by_name[n].id for n in target_group_names}
    managed_group_ids = {groups_by_name[n].id for n in sso_managed_group_names if n in groups_by_name}

    # Add missing memberships
    for gid in target_group_ids:
        if gid not in existing_memberships:
            db.add(UserGroup(user_id=user.id, group_id=gid))

    # Remove stale SSO-managed memberships
    for gid, ug in existing_memberships.items():
        if gid in managed_group_ids and gid not in target_group_ids:
            await db.delete(ug)


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
