"""Admin operation audit log model."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class AdminAuditLog(Base):
    """A historical, append-only record of an admin action.

    ``actor_id`` is deliberately a plain UUID, not a foreign key — this is an audit
    table and must never be blocked by a later deletion of the actor's user row.
    ``actor_email`` is captured at write time so the log stays readable even after that.
    """

    __tablename__ = "admin_audit_logs"
    __table_args__ = (Index("ix_admin_audit_logs_resource", "resource_type", "resource_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    actor_email: Mapped[str] = mapped_column(String(255), nullable=False)
    actor_jti: Mapped[uuid.UUID] = mapped_column(nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    details: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
