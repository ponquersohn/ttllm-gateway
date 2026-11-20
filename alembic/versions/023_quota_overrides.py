"""Add quota_overrides (single-use codes that bypass quotas for a time) and quota.override.

Revision ID: 023
Revises: 022
Create Date: 2026-11-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "023"
down_revision: Union[str, None] = "022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PERMISSION = "quota.override"

_groups = sa.table("groups", sa.column("id", sa.Uuid), sa.column("name", sa.String))
_group_permissions = sa.table(
    "group_permissions",
    sa.column("id", sa.Uuid),
    sa.column("group_id", sa.Uuid),
    sa.column("permission", sa.String),
)


def upgrade() -> None:
    op.create_table(
        "quota_overrides",
        sa.Column("id", sa.Uuid(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("code_hash", sa.String(64), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column("created_by_email", sa.String(255), nullable=True),
        sa.Column("reason", sa.String(1024), nullable=False),
        sa.Column("override_duration_seconds", sa.Integer(), nullable=False),
        sa.Column("code_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("redeemed_by", sa.Uuid(), nullable=True),
        sa.Column("redeemed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("override_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["redeemed_by"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["revoked_by"], ["users.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("code_hash"),
    )
    op.create_index("ix_quota_overrides_code_expires_at", "quota_overrides", ["code_expires_at"])
    op.create_index(
        "ix_quota_overrides_redeemed_by_override_expires_at",
        "quota_overrides",
        ["redeemed_by", "override_expires_at"],
    )

    conn = op.get_bind()
    row = conn.execute(sa.select(_groups.c.id).where(_groups.c.name == "administrators")).fetchone()
    if row is not None:
        existing = conn.execute(
            sa.select(_group_permissions.c.id).where(
                sa.and_(_group_permissions.c.group_id == row[0], _group_permissions.c.permission == PERMISSION)
            )
        ).fetchone()
        if existing is None:
            conn.execute(
                _group_permissions.insert().values(
                    id=sa.text("gen_random_uuid()"), group_id=row[0], permission=PERMISSION
                )
            )


def downgrade() -> None:
    op.get_bind().execute(_group_permissions.delete().where(_group_permissions.c.permission == PERMISSION))
    op.drop_index("ix_quota_overrides_redeemed_by_override_expires_at", table_name="quota_overrides")
    op.drop_index("ix_quota_overrides_code_expires_at", table_name="quota_overrides")
    op.drop_table("quota_overrides")
