"""Redesign audit trail FKs so models/users/groups can be really deleted.

Problem: audit_logs.model_id / audit_logs.user_id (and admin_audit_logs.actor_id) were
mandatory foreign keys into llm_models / users. That permanently blocked deleting a
model or user once it had any history, so "delete" degraded into "deactivate" (is_active
= false) — which in turn meant a deactivated model/user's name/email stayed reserved
forever, since is_active was never actually consulted by the unique constraint. Worse,
`name`/`email` are mutable, so joining audit_logs -> llm_models/users for reporting
silently rewrote history on a rename.

Fix:
- audit_logs.model_id / user_id and admin_audit_logs.actor_id become plain UUID columns
  with no FK at all — kept only for convenience joins against currently-live rows.
- audit_logs gains model_name/model_snapshot and user_email/user_snapshot: a searchable
  field plus a full point-in-time JSON dump, captured at write time, so audit rows are
  self-contained and immune to a later rename or deletion. admin_audit_logs gains
  actor_email for the same reason.
- Existing rows are backfilled from the current llm_models/users tables (still accurate,
  since deletion was never actually possible before this migration).
- The current-state (non-historical) tables that reference users/llm_models/groups
  (model_assignments, group_model_assignments, group_permissions, user_permissions,
  user_groups, refresh_tokens, gateway_tokens) get ON DELETE CASCADE so a real delete
  doesn't get FK-blocked by them.
- llm_models.name / users.email / groups.name keep their existing plain, global unique
  constraints unchanged — is_active stays a pure on/off flag, unrelated to identity or
  deletion. delete_model/delete_user/delete_group now actually remove the row.

Revision ID: 019
Revises: 018
Create Date: 2026-07-03

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "019"
down_revision: Union[str, None] = "018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (table, column, references_table) for FKs recreated with ON DELETE CASCADE.
_CASCADE_FKS = [
    ("model_assignments", "user_id", "users"),
    ("model_assignments", "model_id", "llm_models"),
    ("group_model_assignments", "group_id", "groups"),
    ("group_model_assignments", "model_id", "llm_models"),
    ("group_permissions", "group_id", "groups"),
    ("user_permissions", "user_id", "users"),
    ("user_groups", "user_id", "users"),
    ("user_groups", "group_id", "groups"),
    ("refresh_tokens", "user_id", "users"),
    ("gateway_tokens", "user_id", "users"),
]

# (table, column) for the audit FKs dropped entirely (no replacement).
_AUDIT_FKS_DROPPED = [
    ("audit_logs", "model_id", "llm_models"),
    ("audit_logs", "user_id", "users"),
    ("admin_audit_logs", "actor_id", "users"),
]


def _fk_name(table: str, column: str) -> str:
    return f"{table}_{column}_fkey"


def _model_snapshot(row) -> dict:
    return {
        "id": str(row.id),
        "name": row.name,
        "display_name": row.display_name,
        "provider": row.provider,
        "provider_model_id": row.provider_model_id,
        "config_json": row.config_json,
        "input_cost_per_1k": str(row.input_cost_per_1k),
        "output_cost_per_1k": str(row.output_cost_per_1k),
        "cache_read_cost_per_1k": str(row.cache_read_cost_per_1k),
        "cache_write_cost_per_1k": str(row.cache_write_cost_per_1k),
        "match_pattern": row.match_pattern,
        "is_active": row.is_active,
    }


def _user_snapshot(row) -> dict:
    return {
        "id": str(row.id),
        "name": row.name,
        "email": row.email,
        "identity_provider": row.identity_provider,
        "is_active": row.is_active,
    }


def upgrade() -> None:
    conn = op.get_bind()

    # --- Drop the audit-table FKs entirely (no replacement) ---
    for table, column, _ in _AUDIT_FKS_DROPPED:
        op.drop_constraint(_fk_name(table, column), table, type_="foreignkey")

    # --- Recreate current-state FKs with ON DELETE CASCADE ---
    for table, column, ref_table in _CASCADE_FKS:
        op.drop_constraint(_fk_name(table, column), table, type_="foreignkey")
        op.create_foreign_key(
            _fk_name(table, column), table, ref_table, [column], ["id"], ondelete="CASCADE"
        )

    # --- New snapshot columns ---
    op.add_column("audit_logs", sa.Column("model_name", sa.String(255), nullable=True))
    op.add_column(
        "audit_logs",
        sa.Column("model_snapshot", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column("audit_logs", sa.Column("user_email", sa.String(255), nullable=True))
    op.add_column(
        "audit_logs",
        sa.Column("user_snapshot", postgresql.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.add_column("admin_audit_logs", sa.Column("actor_email", sa.String(255), nullable=True))

    # --- Backfill from the still-live llm_models/users rows (one UPDATE per model/user,
    # not per audit row -- these source tables are small, audit_logs isn't) ---
    # config_json/model_snapshot/user_snapshot are explicitly typed JSONB here (unlike the
    # plain-scalar columns) so values round-trip correctly through the asyncpg dialect --
    # an untyped sa.column() skips JSON encode/decode entirely.
    models_table = sa.table(
        "llm_models",
        sa.column("id"),
        sa.column("name"),
        sa.column("display_name"),
        sa.column("provider"),
        sa.column("provider_model_id"),
        sa.column("config_json", postgresql.JSONB),
        sa.column("input_cost_per_1k"),
        sa.column("output_cost_per_1k"),
        sa.column("cache_read_cost_per_1k"),
        sa.column("cache_write_cost_per_1k"),
        sa.column("match_pattern"),
        sa.column("is_active"),
    )
    users_table = sa.table(
        "users",
        sa.column("id"),
        sa.column("name"),
        sa.column("email"),
        sa.column("identity_provider"),
        sa.column("is_active"),
    )
    audit_logs = sa.table(
        "audit_logs",
        sa.column("model_id"),
        sa.column("model_name"),
        sa.column("model_snapshot", postgresql.JSONB),
        sa.column("user_id"),
        sa.column("user_email"),
        sa.column("user_snapshot", postgresql.JSONB),
    )
    admin_audit_logs = sa.table(
        "admin_audit_logs",
        sa.column("actor_id"),
        sa.column("actor_email"),
    )

    for row in conn.execute(sa.select(models_table)).fetchall():
        conn.execute(
            audit_logs.update()
            .where(audit_logs.c.model_id == row.id)
            .values(model_name=row.name, model_snapshot=_model_snapshot(row))
        )

    for row in conn.execute(sa.select(users_table)).fetchall():
        conn.execute(
            audit_logs.update()
            .where(audit_logs.c.user_id == row.id)
            .values(user_email=row.email, user_snapshot=_user_snapshot(row))
        )
        conn.execute(
            admin_audit_logs.update()
            .where(admin_audit_logs.c.actor_id == row.id)
            .values(actor_email=row.email)
        )

    # Defensive fallback: shouldn't happen (every existing row's model/user still
    # existed until this migration), but never leave a NOT NULL column unbackfillable.
    conn.execute(audit_logs.update().where(audit_logs.c.model_name.is_(None)).values(model_name="(unknown)"))
    conn.execute(audit_logs.update().where(audit_logs.c.user_email.is_(None)).values(user_email="(unknown)"))
    conn.execute(
        admin_audit_logs.update().where(admin_audit_logs.c.actor_email.is_(None)).values(actor_email="(unknown)")
    )

    op.alter_column("audit_logs", "model_name", nullable=False)
    op.alter_column("audit_logs", "user_email", nullable=False)
    op.alter_column("admin_audit_logs", "actor_email", nullable=False)


def downgrade() -> None:
    op.drop_column("admin_audit_logs", "actor_email")
    op.drop_column("audit_logs", "user_snapshot")
    op.drop_column("audit_logs", "user_email")
    op.drop_column("audit_logs", "model_snapshot")
    op.drop_column("audit_logs", "model_name")

    for table, column, ref_table in _CASCADE_FKS:
        op.drop_constraint(_fk_name(table, column), table, type_="foreignkey")
        op.create_foreign_key(_fk_name(table, column), table, ref_table, [column], ["id"])

    for table, column, ref_table in _AUDIT_FKS_DROPPED:
        op.create_foreign_key(_fk_name(table, column), table, ref_table, [column], ["id"])
