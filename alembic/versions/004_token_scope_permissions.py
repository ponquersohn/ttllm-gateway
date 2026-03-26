"""Add permissions column to gateway_tokens.

Revision ID: 004
Revises: 003
Create Date: 2026-04-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "gateway_tokens",
        sa.Column(
            "permissions",
            sa.ARRAY(sa.String()),
            nullable=False,
            server_default="{llm.invoke}",
        ),
    )


def downgrade() -> None:
    op.drop_column("gateway_tokens", "permissions")
