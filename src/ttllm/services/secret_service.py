"""Secret CRUD and runtime config resolution."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.core.secrets import collect_secret_names, decrypt_value, encrypt_value, resolve_config_secrets
from ttllm.models.secret import Secret


def _key() -> str:
    return settings.secrets.encryption_key


async def create_secret(
    db: AsyncSession,
    name: str,
    plaintext_value: str,
    description: str | None = None,
) -> Secret:
    secret = Secret(
        name=name,
        encrypted_value=encrypt_value(plaintext_value, _key()),
        description=description,
    )
    db.add(secret)
    await db.commit()
    await db.refresh(secret)
    return secret


async def get_secret(db: AsyncSession, secret_id: uuid.UUID) -> Secret | None:
    return await db.get(Secret, secret_id)


async def get_secret_by_name(db: AsyncSession, name: str) -> Secret | None:
    result = await db.execute(select(Secret).where(Secret.name == name))
    return result.scalar_one_or_none()


async def list_secrets(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[Secret], int]:
    count_result = await db.execute(select(Secret.id))
    total = len(count_result.all())

    query = (
        select(Secret)
        .offset(offset)
        .limit(limit)
        .order_by(Secret.created_at.desc())
    )
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_secret(
    db: AsyncSession,
    secret_id: uuid.UUID,
    **kwargs,
) -> Secret | None:
    secret = await db.get(Secret, secret_id)
    if not secret:
        return None
    if "plaintext_value" in kwargs:
        secret.encrypted_value = encrypt_value(kwargs["plaintext_value"], _key())
    if "description" in kwargs:
        secret.description = kwargs["description"]
    await db.commit()
    await db.refresh(secret)

    # Invalidate provider cache so stale credentials are not reused
    from ttllm.core.provider import registry
    registry.clear_cache()

    return secret


async def delete_secret(db: AsyncSession, secret_id: uuid.UUID) -> bool:
    secret = await db.get(Secret, secret_id)
    if not secret:
        return False
    await db.delete(secret)
    await db.commit()

    from ttllm.core.provider import registry
    registry.clear_cache()

    return True


async def resolve_model_config(
    db: AsyncSession,
    config_json: dict[str, Any],
) -> dict[str, Any]:
    """Batch-resolve all ``secret://name`` references in a model config dict."""
    names = collect_secret_names(config_json)
    if not names:
        return config_json

    # Batch fetch all referenced secrets in one query
    result = await db.execute(select(Secret).where(Secret.name.in_(names)))
    secrets = {s.name: s for s in result.scalars().all()}

    key = _key()

    def resolver(name: str) -> str | None:
        s = secrets.get(name)
        if s is None:
            return None
        return decrypt_value(s.encrypted_value, key)

    return resolve_config_secrets(config_json, resolver)
