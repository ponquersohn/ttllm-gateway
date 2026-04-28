"""Service for persisting OIDC SSO flow state in the database (encrypted)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.core.secrets import encrypt_value, decrypt_value
from ttllm.models.oidc_state import OidcState

logger = logging.getLogger(__name__)

STATE_TTL = timedelta(minutes=10)


def _encryption_key() -> str:
    key = settings.secrets.encryption_key
    if not key:
        raise RuntimeError("secrets.encryption_key must be configured to use SSO")
    return key


async def store_state(db: AsyncSession, state_key: str, data: dict) -> None:
    """Encrypt and persist OIDC state data."""
    key = _encryption_key()
    encrypted = encrypt_value(json.dumps(data), key)
    row = OidcState(
        state_key=state_key,
        encrypted_data=encrypted,
        expires_at=datetime.now(UTC) + STATE_TTL,
    )
    db.add(row)
    await db.flush()


async def pop_state(db: AsyncSession, state_key: str) -> dict | None:
    """Retrieve, decrypt, and delete OIDC state. Returns None if missing or expired."""
    result = await db.execute(
        select(OidcState).where(OidcState.state_key == state_key)
    )
    row = result.scalar_one_or_none()
    if not row:
        return None
    await db.delete(row)
    await db.flush()
    if row.expires_at < datetime.now(UTC):
        return None
    key = _encryption_key()
    return json.loads(decrypt_value(row.encrypted_data, key))


async def cleanup_expired(db: AsyncSession) -> int:
    """Delete expired state rows. Returns count deleted."""
    result = await db.execute(
        delete(OidcState).where(OidcState.expires_at < datetime.now(UTC))
    )
    await db.flush()
    return result.rowcount
