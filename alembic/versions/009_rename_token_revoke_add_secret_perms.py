"""Rename token.revoke -> token.delete, grant secret.* permissions to administrators.

Revision ID: 009
Revises: 008
Create Date: 2026-04-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SECRET_PERMISSIONS = [
    "secret.view",
    "secret.create",
    "secret.modify",
    "secret.delete",
]


def upgrade() -> None:
    conn = op.get_bind()

    # Rename token.revoke -> token.delete in group_permissions
    conn.execute(
        sa.text(
            "UPDATE group_permissions SET permission = 'token.delete' "
            "WHERE permission = 'token.revoke'"
        )
    )

    # Rename token.revoke -> token.delete in user_permissions
    conn.execute(
        sa.text(
            "UPDATE user_permissions SET permission = 'token.delete' "
            "WHERE permission = 'token.revoke'"
        )
    )

    # Grant secret.* permissions to the administrators group
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
        for perm in SECRET_PERMISSIONS:
            # Avoid duplicate if already present
            existing = conn.execute(
                sa.select(gp_table.c.group_id).where(
                    sa.and_(
                        gp_table.c.group_id == group_id,
                        gp_table.c.permission == perm,
                    )
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


def downgrade() -> None:
    conn = op.get_bind()

    # Revert token.delete -> token.revoke
    conn.execute(
        sa.text(
            "UPDATE group_permissions SET permission = 'token.revoke' "
            "WHERE permission = 'token.delete'"
        )
    )
    conn.execute(
        sa.text(
            "UPDATE user_permissions SET permission = 'token.revoke' "
            "WHERE permission = 'token.delete'"
        )
    )

    # Remove secret.* permissions from administrators
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
        for perm in SECRET_PERMISSIONS:
            conn.execute(
                gp_table.delete().where(
                    sa.and_(
                        gp_table.c.group_id == group_id,
                        gp_table.c.permission == perm,
                    )
                )
            )