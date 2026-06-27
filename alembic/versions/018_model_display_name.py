"""Add display_name column to llm_models.

Optional human-friendly label for models, used by the Anthropic-compatible
model discovery API. Falls back to the name column when NULL.

Revision ID: 018
Revises: 017
Create Date: 2026-06-27

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "018"
down_revision: Union[str, None] = "017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_models", sa.Column("display_name", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("llm_models", "display_name")
