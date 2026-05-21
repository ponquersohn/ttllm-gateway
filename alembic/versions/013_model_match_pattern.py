"""Add match_pattern column to llm_models table.

Revision ID: 013
Revises: 012
Create Date: 2026-05-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_models", sa.Column("match_pattern", sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_models", "match_pattern")
