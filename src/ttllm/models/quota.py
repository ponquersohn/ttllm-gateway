import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class Quota(Base):
    """An admin-defined limit on one measure over an interval. Rules attach it to requests."""

    __tablename__ = "quotas"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    measure: Mapped[str] = mapped_column(String(32))
    limit_value: Mapped[Decimal] = mapped_column(Numeric(38, 18))
    dimensions: Mapped[list[str]] = mapped_column(ARRAY(String(32)), default=list)
    window_type: Mapped[str] = mapped_column(String(16))
    window_unit: Mapped[str | None] = mapped_column(String(16), nullable=True)
    window_duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    enforcement: Mapped[str] = mapped_column(String(16), default="soft")
    on_exceeded: Mapped[dict] = mapped_column(JSONB, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class QuotaBucket(Base):
    """One counter of a quota, for one set of resolved dimension values.

    ``window_start``/``window_end`` are NULL until the first request. The row lock on this
    table is what serializes admission for a bucket.
    """

    __tablename__ = "quota_buckets"
    __table_args__ = (
        UniqueConstraint("quota_id", "bucket_key", name="uq_quota_buckets_quota_id_bucket_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    quota_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("quotas.id", ondelete="CASCADE"))
    bucket_key: Mapped[str] = mapped_column(String(64))
    dimension_values: Mapped[dict] = mapped_column(JSONB, default=dict)
    window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    used: Mapped[Decimal] = mapped_column(Numeric(38, 18), default=Decimal("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class QuotaReservation(Base):
    """An in-flight request's estimated usage.

    There is deliberately no ``reserved`` counter on the bucket: in-flight usage is the sum
    of unexpired rows for the bucket's current window, so it can't drift, and a crashed
    worker's reservation stops counting when it expires.
    """

    __tablename__ = "quota_reservations"
    __table_args__ = (
        Index("ix_quota_reservations_bucket_id_expires_at", "bucket_id", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    bucket_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("quota_buckets.id", ondelete="CASCADE"))
    request_id: Mapped[uuid.UUID] = mapped_column()
    amount: Mapped[Decimal] = mapped_column(Numeric(38, 18))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
