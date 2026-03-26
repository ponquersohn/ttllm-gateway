"""JWT creation and validation. No framework dependencies (uses PyJWT)."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt as pyjwt


@dataclass(frozen=True)
class JWTConfig:
    secret_key: str
    algorithm: str = "HS256"
    issuer: str = "ttllm"


@dataclass(frozen=True)
class TokenPayload:
    sub: uuid.UUID
    permissions: list[str]
    jti: uuid.UUID
    exp: datetime
    iat: datetime


def create_access_token(
    user_id: uuid.UUID,
    permissions: list[str],
    jti: uuid.UUID,
    ttl: timedelta,
    config: JWTConfig,
) -> str:
    """Create a signed JWT access token."""
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "permissions": sorted(permissions),
        "jti": str(jti),
        "iat": now,
        "exp": now + ttl,
        "iss": config.issuer,
    }
    return pyjwt.encode(payload, config.secret_key, algorithm=config.algorithm)


def decode_token(token: str, config: JWTConfig) -> TokenPayload:
    """Decode and validate a JWT, returning a TokenPayload.

    Raises jwt.ExpiredSignatureError, jwt.InvalidTokenError on failure.
    """
    data = pyjwt.decode(
        token,
        config.secret_key,
        algorithms=[config.algorithm],
        issuer=config.issuer,
        options={"require": ["sub", "permissions", "jti", "exp", "iat"]},
    )
    return TokenPayload(
        sub=uuid.UUID(data["sub"]),
        permissions=data["permissions"],
        jti=uuid.UUID(data["jti"]),
        exp=datetime.fromtimestamp(data["exp"], tz=UTC),
        iat=datetime.fromtimestamp(data["iat"], tz=UTC),
    )


def create_refresh_token() -> str:
    """Generate a cryptographically random refresh token."""
    return secrets.token_urlsafe(48)


def hash_refresh_token(token: str) -> str:
    """SHA-256 hash of a refresh token for DB storage."""
    return hashlib.sha256(token.encode()).hexdigest()
