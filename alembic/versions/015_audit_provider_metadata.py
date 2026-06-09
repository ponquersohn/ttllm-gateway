"""Add provider_metadata column to audit_logs table.

Holds the provider's opaque blob (raw usage payload, cost breakdown, latency, stop reason)
alongside the existing metadata_json (client_ip/user_agent).

Revision ID: 015
Revises: 014
Create Date: 2026-06-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "audit_logs",
        sa.Column("provider_metadata", postgresql.JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("audit_logs", "provider_metadata")
