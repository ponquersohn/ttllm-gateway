"""Add device_authorizations table for the OAuth device code flow (RFC 8628).

Revision ID: 021
Revises: 020
Create Date: 2026-11-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "021"
down_revision: Union[str, None] = "020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "device_authorizations",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("device_code_hash", sa.String(64), nullable=False),
        sa.Column("user_code_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("client_name", sa.String(255), nullable=True),
        sa.Column("requester_ip", sa.String(64), nullable=True),
        sa.Column("requester_user_agent", sa.String(512), nullable=True),
        sa.Column("interval_seconds", sa.Integer(), nullable=False),
        sa.Column("last_polled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("device_code_hash"),
        sa.UniqueConstraint("user_code_hash"),
    )
    op.create_index("ix_device_authorizations_expires_at", "device_authorizations", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_device_authorizations_expires_at", table_name="device_authorizations")
    op.drop_table("device_authorizations")
