"""Add rules table for API-configurable rules engine.

Revision ID: 016
Revises: 015
Create Date: 2026-05-27

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RULE_PERMISSIONS = [
    "rule.view",
    "rule.create",
    "rule.modify",
    "rule.delete",
]


def _grant_rule_permissions(conn) -> None:
    """Grant rule.* to the administrators group (idempotent), mirroring 009."""
    groups_table = sa.table("groups", sa.column("id", sa.Uuid), sa.column("name", sa.String))
    gp_table = sa.table(
        "group_permissions",
        sa.column("id", sa.Uuid),
        sa.column("group_id", sa.Uuid),
        sa.column("permission", sa.String),
    )
    row = conn.execute(
        sa.select(groups_table.c.id).where(groups_table.c.name == "administrators")
    ).fetchone()
    if row is None:
        return
    group_id = row[0]
    for perm in RULE_PERMISSIONS:
        existing = conn.execute(
            sa.select(gp_table.c.group_id).where(
                sa.and_(gp_table.c.group_id == group_id, gp_table.c.permission == perm)
            )
        ).fetchone()
        if existing is None:
            conn.execute(
                gp_table.insert().values(
                    id=sa.text("gen_random_uuid()"),
                    group_id=group_id,
                    permission=perm,
                )
            )


def upgrade() -> None:
    op.create_table(
        "rules",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("weight", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("conditions", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("action", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_rules_enabled", "rules", ["enabled"])
    op.create_index("ix_rules_weight", "rules", ["weight"])

    _grant_rule_permissions(op.get_bind())


def downgrade() -> None:
    conn = op.get_bind()
    gp_table = sa.table(
        "group_permissions",
        sa.column("group_id", sa.Uuid),
        sa.column("permission", sa.String),
    )
    groups_table = sa.table("groups", sa.column("id", sa.Uuid), sa.column("name", sa.String))
    row = conn.execute(
        sa.select(groups_table.c.id).where(groups_table.c.name == "administrators")
    ).fetchone()
    if row is not None:
        group_id = row[0]
        for perm in RULE_PERMISSIONS:
            conn.execute(
                gp_table.delete().where(
                    sa.and_(gp_table.c.group_id == group_id, gp_table.c.permission == perm)
                )
            )

    op.drop_index("ix_rules_weight", table_name="rules")
    op.drop_index("ix_rules_enabled", table_name="rules")
    op.drop_table("rules")
