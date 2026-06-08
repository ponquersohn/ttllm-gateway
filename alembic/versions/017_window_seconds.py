"""Add window_seconds to token_limits for configurable window duration.

Revision ID: 017
Revises: 016
Create Date: 2026-06-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("token_limits", sa.Column("window_seconds", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("token_limits", "window_seconds")
