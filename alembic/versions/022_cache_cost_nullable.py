"""Make llm_models cache prices nullable; NULL bills cache tokens at the input rate.

Until now the cache read/write prices defaulted to 0, so a model whose cache prices were
never configured billed every cached token at $0, and cost quotas could be sidestepped
by marking a large prompt for caching. NULL now means "not set": the provider bills those
tokens at the model's input rate instead.

Existing 0s are converted to NULL. 0 was the column default and is almost never a
deliberate price; a model that really is free has an input price of 0 too, so the
fallback still bills 0. An explicit 0 set after this migration is kept as 0.

Revision ID: 022
Revises: 021
Create Date: 2026-11-16

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "022"
down_revision: Union[str, None] = "021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_COLUMNS = ("cache_read_cost_per_1k", "cache_write_cost_per_1k")


def upgrade() -> None:
    for column in _COLUMNS:
        op.alter_column(
            "llm_models",
            column,
            existing_type=sa.Numeric(10, 6),
            nullable=True,
            server_default=None,
        )
        op.execute(f"UPDATE llm_models SET {column} = NULL WHERE {column} = 0")


def downgrade() -> None:
    for column in _COLUMNS:
        op.execute(f"UPDATE llm_models SET {column} = 0 WHERE {column} IS NULL")
        op.alter_column(
            "llm_models",
            column,
            existing_type=sa.Numeric(10, 6),
            nullable=False,
            server_default="0",
        )
