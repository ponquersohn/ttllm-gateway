"""Seed a default admin user so the system is usable after first migration.

Revision ID: 006
Revises: 005
Create Date: 2026-04-08

The admin password is read from TTLLM_ADMIN_PASSWORD (defaults to "admin").
"""

import os
from typing import Sequence, Union

import bcrypt
import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

DEFAULT_EMAIL = "admin@localhost"


def upgrade() -> None:
    conn = op.get_bind()

    # Skip if an admin user already exists (re-run safety).
    existing = conn.execute(
        sa.text("SELECT id FROM users WHERE email = :email"),
        {"email": DEFAULT_EMAIL},
    ).fetchone()
    if existing is not None:
        return

    import uuid

    user_id = uuid.uuid4()
    password = os.environ.get("TTLLM_ADMIN_PASSWORD", "admin")
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    conn.execute(
        sa.text(
            "INSERT INTO users (id, name, email, password_hash, is_active, created_at, updated_at) "
            "VALUES (:id, :name, :email, :pw, true, now(), now())"
        ),
        {
            "id": user_id,
            "name": "Admin",
            "email": DEFAULT_EMAIL,
            "password_hash": password_hash,
        },
    )

    # Add to the administrators group created in migration 002.
    admin_group = conn.execute(
        sa.text("SELECT id FROM groups WHERE name = 'administrators'")
    ).fetchone()
    if admin_group is not None:
        conn.execute(
            sa.text(
                "INSERT INTO user_groups (id, user_id, group_id) "
                "VALUES (:id, :uid, :gid)"
            ),
            {"id": uuid.uuid4(), "uid": user_id, "gid": admin_group[0]},
        )

    # Also add to the default users group so the admin can invoke LLMs.
    users_group = conn.execute(
        sa.text("SELECT id FROM groups WHERE name = 'users'")
    ).fetchone()
    if users_group is not None:
        conn.execute(
            sa.text(
                "INSERT INTO user_groups (id, user_id, group_id) "
                "VALUES (:id, :uid, :gid)"
            ),
            {"id": uuid.uuid4(), "uid": user_id, "gid": users_group[0]},
        )


def downgrade() -> None:
    conn = op.get_bind()
    user = conn.execute(
        sa.text("SELECT id FROM users WHERE email = :email"),
        {"email": DEFAULT_EMAIL},
    ).fetchone()
    if user is not None:
        user_id = user[0]
        conn.execute(
            sa.text("DELETE FROM user_groups WHERE user_id = :uid"),
            {"uid": user_id},
        )
        conn.execute(
            sa.text("DELETE FROM users WHERE id = :uid"),
            {"uid": user_id},
        )
