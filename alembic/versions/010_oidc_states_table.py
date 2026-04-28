"""Add oidc_states table for encrypted OIDC flow state.

Revision ID: 010
Revises: 009
Create Date: 2026-04-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "oidc_states",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("state_key", sa.String(64), nullable=False),
        sa.Column("encrypted_data", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("state_key"),
    )
    op.create_index("ix_oidc_states_state_key", "oidc_states", ["state_key"])
    op.create_index("ix_oidc_states_expires_at", "oidc_states", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_oidc_states_expires_at", table_name="oidc_states")
    op.drop_index("ix_oidc_states_state_key", table_name="oidc_states")
    op.drop_table("oidc_states")
