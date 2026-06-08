"""SQLAlchemy models for token quota limits and usage counters."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class LimitScope(str, enum.Enum):
    USER = "user"
    GROUP = "group"
    GLOBAL = "global"


class WindowKind(str, enum.Enum):
    FIVE_H = "5h"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# Persist enum *values* ("user", "5h") to match the Postgres enum types created
# in migration 015, not the Python member *names* ("USER", "FIVE_H").
def _enum_values(e: type[enum.Enum]) -> list[str]:
    return [member.value for member in e]


class TokenLimit(Base):
    __tablename__ = "token_limits"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scope: Mapped[LimitScope] = mapped_column(
        Enum(LimitScope, name="limit_scope", values_callable=_enum_values), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    group_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"), nullable=True, index=True
    )
    window_kind: Mapped[WindowKind] = mapped_column(
        Enum(WindowKind, name="window_kind", values_callable=_enum_values), nullable=False
    )
    token_cap: Mapped[int] = mapped_column(BigInteger, nullable=False)
    window_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UsageCounter(Base):
    __tablename__ = "usage_counters"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    window_kind: Mapped[WindowKind] = mapped_column(
        Enum(WindowKind, name="window_kind", values_callable=_enum_values), primary_key=True
    )
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tokens_used: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
