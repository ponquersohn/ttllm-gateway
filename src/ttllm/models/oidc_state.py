"""OIDC state ORM model: stores encrypted SSO flow state in the database."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class OidcState(Base):
    __tablename__ = "oidc_states"
    # A separate UniqueConstraint (Postgres auto-names it oidc_states_state_key_key) plus
    # a plain index=True below, matching the two distinct objects migration 010 actually
    # created -- not the single combined unique index that `unique=True, index=True`
    # inline would declare.
    __table_args__ = (UniqueConstraint("state_key"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    state_key: Mapped[str] = mapped_column(String(64), index=True)
    encrypted_data: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
