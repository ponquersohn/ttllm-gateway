import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ttllm.models import Base


class LLMModel(Base):
    __tablename__ = "llm_models"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    provider: Mapped[str] = mapped_column(String(50))
    provider_model_id: Mapped[str] = mapped_column(String(255))
    config_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    input_cost_per_1k: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), default=Decimal("0")
    )
    output_cost_per_1k: Mapped[Decimal] = mapped_column(
        Numeric(10, 6), default=Decimal("0")
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    assignments: Mapped[list["ModelAssignment"]] = relationship(back_populates="model")
    group_assignments: Mapped[list["GroupModelAssignment"]] = relationship(back_populates="model")


class ModelAssignment(Base):
    __tablename__ = "model_assignments"
    __table_args__ = (UniqueConstraint("user_id", "model_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), nullable=False)
    model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_models.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="model_assignments")
    model: Mapped["LLMModel"] = relationship(back_populates="assignments")


class GroupModelAssignment(Base):
    __tablename__ = "group_model_assignments"
    __table_args__ = (UniqueConstraint("group_id", "model_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("groups.id"), nullable=False
    )
    model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("llm_models.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    model: Mapped["LLMModel"] = relationship(back_populates="group_assignments")
