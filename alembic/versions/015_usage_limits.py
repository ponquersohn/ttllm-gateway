"""Add token_limits and usage_counters tables.

Revision ID: 015
Revises: 014
Create Date: 2026-06-08
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "015"
down_revision: Union[str, None] = "014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "token_limits",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("scope", sa.Enum("user", "group", "global", name="limit_scope"), nullable=False),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("group_id", sa.Uuid(), sa.ForeignKey("groups.id", ondelete="CASCADE"), nullable=True),
        sa.Column("window_kind", sa.Enum("5h", "weekly", "monthly", name="window_kind"), nullable=False),
        sa.Column("token_cap", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_token_limits_user_id", "token_limits", ["user_id"])
    op.create_index("ix_token_limits_group_id", "token_limits", ["group_id"])

    op.create_table(
        "usage_counters",
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window_kind", sa.Enum("5h", "weekly", "monthly", name="window_kind"), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tokens_used", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("cooldown_until", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("user_id", "window_kind"),
    )


def downgrade() -> None:
    op.drop_table("usage_counters")
    op.drop_index("ix_token_limits_group_id", table_name="token_limits")
    op.drop_index("ix_token_limits_user_id", table_name="token_limits")
    op.drop_table("token_limits")
    sa.Enum(name="limit_scope").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="window_kind").drop(op.get_bind(), checkfirst=True)
