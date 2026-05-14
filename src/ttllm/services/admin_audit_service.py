"""Admin audit log writing."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.admin_audit import AdminAuditLog


async def log(
    db: AsyncSession,
    *,
    actor_id: uuid.UUID,
    actor_jti: uuid.UUID,
    action: str,
    resource_type: str,
    resource_id: uuid.UUID,
    details: dict[str, Any] | None = None,
) -> None:
    """Write a single admin audit log entry."""
    db.add(
        AdminAuditLog(
            actor_id=actor_id,
            actor_jti=actor_jti,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
        )
    )
    await db.commit()
