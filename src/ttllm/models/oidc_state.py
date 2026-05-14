"""OIDC state ORM model: stores encrypted SSO flow state in the database."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ttllm.models import Base


class OidcState(Base):
    __tablename__ = "oidc_states"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    state_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    encrypted_data: Mapped[str] = mapped_column(Text)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
