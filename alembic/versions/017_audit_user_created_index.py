"""Add composite (user_id, created_at) index on audit_logs.

Speeds up the rules engine's moving-window quota aggregates, which filter by
user_id and a trailing created_at range. The existing single-column indexes on
user_id and created_at can't serve the combined predicate as efficiently.

Revision ID: 017
Revises: 016
Create Date: 2026-06-12

"""

from typing import Sequence, Union

from alembic import op

revision: str = "017"
down_revision: Union[str, None] = "016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_audit_logs_user_id_created_at",
        "audit_logs",
        ["user_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_user_id_created_at", table_name="audit_logs")
