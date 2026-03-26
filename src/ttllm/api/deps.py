"""FastAPI dependency injection for JWT auth, permission checking, and DB sessions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.core.jwt import JWTConfig, TokenPayload, decode_token
from ttllm.db import async_session_factory
from ttllm.models.user import User
from ttllm.services import auth_service


@dataclass
class AuthContext:
    user: User
    permissions: set[str]
    jti: uuid.UUID


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session


async def _authenticate(token: str, db: AsyncSession) -> AuthContext:
    """Shared JWT validation and user loading."""
    jwt_config = JWTConfig(
        secret_key=settings.auth.jwt.secret_key,
        algorithm=settings.auth.jwt.algorithm,
    )

    try:
        payload: TokenPayload = decode_token(token, jwt_config)
    except Exception:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid or expired token"},
        )

    # For tokens created via the token API, verify the DB record is still active.
    # Login-issued tokens won't have a record and skip this check.
    if await auth_service.token_exists(db, payload.jti):
        if not await auth_service.validate_token(db, payload.jti):
            raise HTTPException(
                status_code=401,
                detail={"type": "authentication_error", "message": "Token has been revoked"},
            )

    # Load user
    user = await db.get(User, payload.sub)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=403,
            detail={"type": "permission_error", "message": "User deactivated"},
        )

    return AuthContext(
        user=user,
        permissions=set(payload.permissions),
        jti=payload.jti,
    )


async def get_authenticated(
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db),
) -> AuthContext:
    """Authenticate via Authorization: Bearer <jwt> (management APIs)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Missing or invalid Authorization header"},
        )
    return await _authenticate(authorization[7:], db)


def require_permission(*perms: str, auth_dep=get_authenticated):
    """Dependency factory: returns a dependency that checks the caller has all listed permissions."""
    async def checker(
        ctx: AuthContext = Depends(auth_dep),
    ) -> AuthContext:
        registry = auth_service.get_permission_registry()
        for p in perms:
            if not registry.check(ctx.permissions, p):
                raise HTTPException(
                    status_code=403,
                    detail={
                        "type": "permission_error",
                        "message": f"Missing required permission: {p}",
                    },
                )
        return ctx
    return checker


DB = Annotated[AsyncSession, Depends(get_db)]
