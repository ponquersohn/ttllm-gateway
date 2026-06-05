"""Add cache pricing columns to llm_models table.

Revision ID: 014
Revises: 013
Create Date: 2026-05-31

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_models",
        sa.Column("cache_read_cost_per_1k", sa.Numeric(10, 6), nullable=False, server_default="0"),
    )
    op.add_column(
        "llm_models",
        sa.Column("cache_write_cost_per_1k", sa.Numeric(10, 6), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("llm_models", "cache_write_cost_per_1k")
    op.drop_column("llm_models", "cache_read_cost_per_1k")
