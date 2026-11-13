"""Device authorization ORM model: pending RFC 8628 device-code logins."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class DeviceAuthorization(Base):
    __tablename__ = "device_authorizations"
    __table_args__ = (
        UniqueConstraint("device_code_hash"),
        UniqueConstraint("user_code_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Only SHA-256 hashes of both codes are stored.
    device_code_hash: Mapped[str] = mapped_column(String(64))
    user_code_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    client_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    requester_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    requester_user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    interval_seconds: Mapped[int] = mapped_column(Integer)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
