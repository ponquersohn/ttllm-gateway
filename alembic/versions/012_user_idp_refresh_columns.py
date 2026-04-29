"""Add idp_refresh_token and last_role_sync_at columns to users table.

Revision ID: 012
Revises: 011
Create Date: 2026-04-28

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("users", sa.Column("idp_refresh_token", sa.Text(), nullable=True))
    op.add_column("users", sa.Column("last_role_sync_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_role_sync_at")
    op.drop_column("users", "idp_refresh_token")
