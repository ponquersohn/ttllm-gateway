"""Grant quota.manage permission to administrators group.

Revision ID: 016
Revises: 015
Create Date: 2026-06-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "016"
down_revision: Union[str, None] = "015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

QUOTA_PERMISSION = "quota.manage"


def upgrade() -> None:
    conn = op.get_bind()

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
    if row is not None:
        group_id = row[0]
        existing = conn.execute(
            sa.select(gp_table.c.group_id).where(
                sa.and_(
                    gp_table.c.group_id == group_id,
                    gp_table.c.permission == QUOTA_PERMISSION,
                )
            )
        ).fetchone()
        if existing is None:
            conn.execute(
                gp_table.insert().values(
                    id=sa.text("gen_random_uuid()"),
                    group_id=group_id,
                    permission=QUOTA_PERMISSION,
                )
            )


def downgrade() -> None:
    conn = op.get_bind()

    groups_table = sa.table("groups", sa.column("id", sa.Uuid), sa.column("name", sa.String))
    gp_table = sa.table(
        "group_permissions",
        sa.column("group_id", sa.Uuid),
        sa.column("permission", sa.String),
    )

    row = conn.execute(
        sa.select(groups_table.c.id).where(groups_table.c.name == "administrators")
    ).fetchone()
    if row is not None:
        group_id = row[0]
        conn.execute(
            gp_table.delete().where(
                sa.and_(
                    gp_table.c.group_id == group_id,
                    gp_table.c.permission == QUOTA_PERMISSION,
                )
            )
        )
