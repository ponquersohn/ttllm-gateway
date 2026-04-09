"""Grant server.status permission to users and administrators groups.

Revision ID: 008
Revises: 007
Create Date: 2026-04-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PERMISSION = "server.status"
GROUP_NAMES = ("users", "administrators")


def upgrade() -> None:
    conn = op.get_bind()
    groups_table = sa.table("groups", sa.column("id", sa.Uuid), sa.column("name", sa.String))
    gp_table = sa.table(
        "group_permissions",
        sa.column("id", sa.Uuid),
        sa.column("group_id", sa.Uuid),
        sa.column("permission", sa.String),
    )

    for group_name in GROUP_NAMES:
        row = conn.execute(
            sa.select(groups_table.c.id).where(groups_table.c.name == group_name)
        ).fetchone()
        if row is None:
            continue
        group_id = row[0]
        conn.execute(
            gp_table.insert().values(
                id=sa.text("gen_random_uuid()"),
                group_id=group_id,
                permission=PERMISSION,
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

    for group_name in GROUP_NAMES:
        row = conn.execute(
            sa.select(groups_table.c.id).where(groups_table.c.name == group_name)
        ).fetchone()
        if row is None:
            continue
        group_id = row[0]
        conn.execute(
            gp_table.delete().where(
                sa.and_(
                    gp_table.c.group_id == group_id,
                    gp_table.c.permission == PERMISSION,
                )
            )
        )
