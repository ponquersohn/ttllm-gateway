"""Replace group_roles with direct group_permissions and user_permissions.

Revision ID: 005
Revises: 004
Create Date: 2026-04-04

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Hardcoded role→permissions mapping for self-contained data migration.
ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin": [
        "user.view", "user.create", "user.modify", "user.delete",
        "group.view", "group.create", "group.modify", "group.delete",
        "model.view", "model.create", "model.modify", "model.delete", "model.assign",
        "token.create", "token.revoke",
        "audit.view", "usage.view",
    ],
    "viewer": ["user.view", "group.view", "model.view", "audit.view", "usage.view"],
    "user-manager": [
        "user.view", "user.create", "user.modify", "user.delete",
        "token.create", "token.revoke",
    ],
    "model-manager": [
        "model.view", "model.create", "model.modify", "model.delete", "model.assign",
    ],
    "llm-user": ["llm.invoke"],
}


def upgrade() -> None:
    op.create_table(
        "group_permissions",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("group_id", sa.Uuid(), sa.ForeignKey("groups.id"), nullable=False),
        sa.Column("permission", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("group_id", "permission"),
    )

    op.create_table(
        "user_permissions",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("permission", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", "permission"),
    )

    # Expand group_roles → group_permissions
    conn = op.get_bind()
    rows = conn.execute(sa.text("SELECT group_id, role_name FROM group_roles")).fetchall()

    seen: set[tuple[str, str]] = set()
    for group_id, role_name in rows:
        perms = ROLE_PERMISSIONS.get(role_name)
        if perms is None:
            raise RuntimeError(
                f"Unknown role '{role_name}' in group_roles for group {group_id}. "
                f"Known roles: {list(ROLE_PERMISSIONS)}"
            )
        for perm in perms:
            key = (str(group_id), perm)
            if key not in seen:
                seen.add(key)
                conn.execute(
                    sa.text(
                        "INSERT INTO group_permissions (id, group_id, permission) "
                        "VALUES (gen_random_uuid(), :gid, :perm)"
                    ),
                    {"gid": group_id, "perm": perm},
                )

    op.drop_table("group_roles")


def downgrade() -> None:
    op.create_table(
        "group_roles",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("group_id", sa.Uuid(), sa.ForeignKey("groups.id"), nullable=False),
        sa.Column("role_name", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("group_id", "role_name"),
    )
    # Best-effort: no reverse mapping attempted — data is lossy.
    op.drop_table("user_permissions")
    op.drop_table("group_permissions")
