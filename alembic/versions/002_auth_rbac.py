"""Auth RBAC: groups, gateway tokens, refresh tokens; drop api_keys and is_admin.

Revision ID: 002
Revises: 001
Create Date: 2026-03-30

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- New tables ---

    op.create_table(
        "groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "group_roles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.Column("role_name", sa.String(100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["group_id"], ["groups.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("group_id", "role_name"),
    )

    op.create_table(
        "user_groups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("group_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["group_id"], ["groups.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "group_id"),
    )

    op.create_table(
        "gateway_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_hash", sa.String(128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    # --- Modify users table ---

    op.add_column("users", sa.Column("password_hash", sa.String(255), nullable=True))
    op.add_column("users", sa.Column("identity_provider", sa.String(100), nullable=True))
    op.add_column("users", sa.Column("external_id", sa.String(255), nullable=True))

    # --- Data migration: is_admin -> administrators group ---

    conn = op.get_bind()

    import uuid

    admin_group_id = uuid.uuid4()
    admin_role_id = uuid.uuid4()

    users_group_id = uuid.uuid4()
    users_role_id = uuid.uuid4()

    conn.execute(
        sa.text(
            "INSERT INTO groups (id, name, description) VALUES (:id, :name, :desc)"
        ),
        {"id": admin_group_id, "name": "administrators", "desc": "Built-in admin group"},
    )
    conn.execute(
        sa.text(
            "INSERT INTO group_roles (id, group_id, role_name) VALUES (:id, :gid, :role)"
        ),
        {"id": admin_role_id, "gid": admin_group_id, "role": "admin"},
    )
    conn.execute(
        sa.text(
            "INSERT INTO groups (id, name, description) VALUES (:id, :name, :desc)"
        ),
        {"id": users_group_id, "name": "users", "desc": "Default user group"},
    )
    conn.execute(
        sa.text(
            "INSERT INTO group_roles (id, group_id, role_name) VALUES (:id, :gid, :role)"
        ),
        {"id": users_role_id, "gid": users_group_id, "role": "llm-user"},
    )

    # Move is_admin=true users into the administrators group
    admin_users = conn.execute(
        sa.text("SELECT id FROM users WHERE is_admin = true")
    ).fetchall()
    for row in admin_users:
        conn.execute(
            sa.text(
                "INSERT INTO user_groups (id, user_id, group_id) VALUES (:id, :uid, :gid)"
            ),
            {"id": uuid.uuid4(), "uid": row[0], "gid": admin_group_id},
        )

    # --- Drop old columns and tables ---

    op.drop_column("users", "is_admin")
    op.drop_index("ix_api_keys_key_prefix", table_name="api_keys")
    op.drop_table("api_keys")


def downgrade() -> None:
    # Re-create api_keys
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("key_hash", sa.String(128), nullable=False),
        sa.Column("label", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_api_keys_key_prefix", "api_keys", ["key_prefix"])

    # Re-add is_admin
    op.add_column(
        "users",
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default="false"),
    )

    # Drop new columns
    op.drop_column("users", "external_id")
    op.drop_column("users", "identity_provider")
    op.drop_column("users", "password_hash")

    # Drop new tables
    op.drop_table("refresh_tokens")
    op.drop_table("gateway_tokens")
    op.drop_table("user_groups")
    op.drop_table("group_roles")
    op.drop_table("groups")
