"""Admin audit log writing."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.admin_audit import AdminAuditLog
from ttllm.models.user import User


async def log(
    db: AsyncSession,
    *,
    actor: User,
    actor_jti: uuid.UUID,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID,
    details: dict[str, Any] | None = None,
) -> None:
    """Write a single admin audit log entry.

    ``actor_email`` is captured from *actor* at write time, so this log stays readable
    even after the actor's user row is later renamed or deleted — ``actor_id`` carries
    no FK and is kept only for convenience joins against currently-live rows.
    """
    db.add(
        AdminAuditLog(
            actor_id=actor.id,
            actor_email=actor.email,
            actor_jti=actor_jti,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )
    )
    await db.commit()
