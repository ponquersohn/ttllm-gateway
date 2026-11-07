"""Add interval-based quota tables and disable rules using the old quota condition.

The ``quota`` rule condition (a rolling window aggregated from audit_logs on every request)
is gone, replaced by quotas, quota_buckets and quota_reservations. Old rules are not
converted: rolling windows have no exact interval equivalent, so every rule containing a
``quota`` condition is disabled, marked in its description, and logged. Those limits are
NOT enforced until an administrator recreates them as quotas.

Downgrade drops the quota tables and disables rules that use ``apply_quota`` or ``subject``,
which the old code can't run. Rules disabled by this upgrade stay disabled, and quota
definitions and usage are lost.

Revision ID: 020
Revises: 019
Create Date: 2026-11-06

"""

import logging
from typing import Any, Callable, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "020"
down_revision: Union[str, None] = "019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")

QUOTA_PERMISSIONS = ["quota.view", "quota.create", "quota.modify", "quota.delete"]

_DISABLED_MARKER = "[Disabled by migration 020: recreate this limit as a quota] "
_DESCRIPTION_MAX = 1024

# Typed JSONB so values decode through the asyncpg dialect (see migration 019).
_rules = sa.table(
    "rules",
    sa.column("id", sa.Uuid),
    sa.column("name", sa.String),
    sa.column("description", sa.String),
    sa.column("enabled", sa.Boolean),
    sa.column("conditions", postgresql.JSONB),
    sa.column("action", postgresql.JSONB),
)


def _any_condition(group: Any, predicate: Callable[[dict], bool]) -> bool:
    """Whether any leaf condition at any nesting depth satisfies ``predicate``."""
    if not isinstance(group, dict):
        return False
    for item in group.get("conditions", []):
        if isinstance(item, dict) and "conditions" in item:
            if _any_condition(item, predicate):
                return True
        elif isinstance(item, dict) and predicate(item):
            return True
    return False


def _grant_quota_permissions(conn) -> None:
    """Grant quota.* to the administrators group (idempotent), mirroring 016."""
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
    for perm in QUOTA_PERMISSIONS:
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


def _disable_legacy_quota_rules(conn) -> None:
    rows = conn.execute(
        sa.select(_rules.c.id, _rules.c.name, _rules.c.description, _rules.c.conditions)
    ).fetchall()
    for rule_id, name, description, conditions in rows:
        if not _any_condition(conditions, lambda c: c.get("type") == "quota"):
            continue
        marked = (_DISABLED_MARKER + (description or "")).strip()[:_DESCRIPTION_MAX]
        conn.execute(
            _rules.update().where(_rules.c.id == rule_id).values(enabled=False, description=marked)
        )
        logger.warning(
            "Rule '%s' used the removed quota condition and has been disabled; "
            "its limit is NOT enforced until it is recreated as a quota.",
            name,
        )


def upgrade() -> None:
    op.create_table(
        "quotas",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.String(1024), nullable=True),
        sa.Column("measure", sa.String(32), nullable=False),
        sa.Column("limit_value", sa.Numeric(38, 18), nullable=False),
        sa.Column("dimensions", postgresql.ARRAY(sa.String(32)), nullable=False),
        sa.Column("window_type", sa.String(16), nullable=False),
        sa.Column("window_unit", sa.String(16), nullable=True),
        sa.Column("window_duration_seconds", sa.Integer, nullable=True),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("enforcement", sa.String(16), nullable=False),
        sa.Column("on_exceeded", postgresql.JSONB, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "quota_buckets",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("quota_id", sa.Uuid, sa.ForeignKey("quotas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("bucket_key", sa.String(64), nullable=False),
        sa.Column("dimension_values", postgresql.JSONB, nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("used", sa.Numeric(38, 18), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("quota_id", "bucket_key", name="uq_quota_buckets_quota_id_bucket_key"),
    )
    op.create_table(
        "quota_reservations",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("bucket_id", sa.Uuid, sa.ForeignKey("quota_buckets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("request_id", sa.Uuid, nullable=False),
        sa.Column("amount", sa.Numeric(38, 18), nullable=False),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_quota_reservations_bucket_id_expires_at",
        "quota_reservations",
        ["bucket_id", "expires_at"],
    )

    conn = op.get_bind()
    _grant_quota_permissions(conn)
    _disable_legacy_quota_rules(conn)


def downgrade() -> None:
    conn = op.get_bind()
    rows = conn.execute(sa.select(_rules.c.id, _rules.c.conditions, _rules.c.action)).fetchall()
    for rule_id, conditions, action in rows:
        uses_new_features = (action or {}).get("type") == "apply_quota" or _any_condition(
            conditions, lambda c: c.get("type") == "subject"
        )
        if uses_new_features:
            conn.execute(_rules.update().where(_rules.c.id == rule_id).values(enabled=False))

    gp_table = sa.table("group_permissions", sa.column("permission", sa.String))
    conn.execute(gp_table.delete().where(gp_table.c.permission.in_(QUOTA_PERMISSIONS)))

    op.drop_table("quota_reservations")
    op.drop_table("quota_buckets")
    op.drop_table("quotas")
