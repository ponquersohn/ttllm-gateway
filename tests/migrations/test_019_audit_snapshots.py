"""Behavior tests specific to migration 019 (audit trail redesign: drop the audit_logs/
admin_audit_logs FKs, add model_name/model_snapshot + user_email/user_snapshot, cascade
the current-state tables). These mirror exactly what was checked by hand against a real
Postgres while writing that migration -- including the JSONB double-encoding bug that
checking by hand actually caught."""

from __future__ import annotations

import uuid

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tests.migrations.conftest import run_sql

pytestmark = pytest.mark.migrations

MODEL_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "22222222-2222-2222-2222-222222222222"


def _seed_pre_019_data(db_url: str) -> None:
    """Seed one model/user/audit_log/admin_audit_log/model_assignment row at revision 018
    (before the snapshot columns and FK removal), representative of production data at
    the point this migration runs."""

    async def _run(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO llm_models (id, name, display_name, provider, provider_model_id, config_json, is_active)
                    VALUES (:id, 'claude-x', 'Claude X', 'bedrock', 'anthropic.claude-x',
                            '{"region": "us-east-1", "aws_access_key_id": "secret://foo"}'::jsonb, true)
                    """
                ),
                {"id": MODEL_ID},
            )
            await conn.execute(
                text("INSERT INTO users (id, name, email, is_active) VALUES (:id, 'Alice', 'alice@example.com', true)"),
                {"id": USER_ID},
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO audit_logs (id, user_id, model_id, request_id, input_tokens, output_tokens, total_cost)
                    VALUES (:id, :user_id, :model_id, :request_id, 10, 20, '0.05')
                    """
                ),
                {"id": str(uuid.uuid4()), "user_id": USER_ID, "model_id": MODEL_ID, "request_id": str(uuid.uuid4())},
            )
            await conn.execute(
                text(
                    """
                    INSERT INTO admin_audit_logs (id, actor_id, actor_jti, action, resource_type, resource_id)
                    VALUES (:id, :actor_id, :jti, 'model.create', 'model', :resource_id)
                    """
                ),
                {"id": str(uuid.uuid4()), "actor_id": USER_ID, "jti": str(uuid.uuid4()), "resource_id": MODEL_ID},
            )
            await conn.execute(
                text("INSERT INTO model_assignments (id, user_id, model_id) VALUES (:id, :user_id, :model_id)"),
                {"id": str(uuid.uuid4()), "user_id": USER_ID, "model_id": MODEL_ID},
            )

    run_sql(db_url, _run)


def test_backfill_populates_snapshot_and_name(alembic_cfg, fresh_db_url):
    """The regression test for the actual bug this migration hit: an untyped sa.column()
    on a JSONB field skips asyncpg's JSON encode/decode entirely, so a naive backfill
    embeds config_json as a JSON-encoded *string* inside model_snapshot instead of a
    nested object. Assert it comes back as a real dict, not str."""
    command.upgrade(alembic_cfg, "018")
    _seed_pre_019_data(fresh_db_url)
    command.upgrade(alembic_cfg, "019")

    async def _fetch(engine):
        async with engine.connect() as conn:
            row = (await conn.execute(text("SELECT model_name, model_snapshot, user_email, user_snapshot FROM audit_logs"))).one()
            actor = (await conn.execute(text("SELECT actor_email FROM admin_audit_logs"))).one()
            return row, actor

    row, actor = run_sql(fresh_db_url, _fetch)
    assert row.model_name == "claude-x"
    assert row.user_email == "alice@example.com"
    assert isinstance(row.model_snapshot, dict)
    assert isinstance(row.model_snapshot["config_json"], dict), "config_json should be a nested object, not a JSON string"
    assert row.model_snapshot["config_json"]["aws_access_key_id"] == "secret://foo"
    assert isinstance(row.user_snapshot, dict)
    assert row.user_snapshot["email"] == "alice@example.com"
    assert actor.actor_email == "alice@example.com"


def test_cascade_and_audit_survives_hard_delete(alembic_cfg, fresh_db_url):
    """After a real DELETE of the model/user, model_assignments must cascade away, and
    the audit rows must survive untouched (name/snapshot intact, id columns just orphaned,
    no FK error) -- the entire point of this migration."""
    command.upgrade(alembic_cfg, "018")
    _seed_pre_019_data(fresh_db_url)
    command.upgrade(alembic_cfg, "019")

    async def _delete_and_check(engine):
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM llm_models WHERE id = :id"), {"id": MODEL_ID})
            await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": USER_ID})
        async with engine.connect() as conn:
            assignments = (await conn.execute(text("SELECT count(*) FROM model_assignments"))).scalar()
            audit_row = (await conn.execute(text("SELECT model_name, user_email, model_id, user_id FROM audit_logs"))).one()
            return assignments, audit_row

    assignments, audit_row = run_sql(fresh_db_url, _delete_and_check)
    assert assignments == 0, "model_assignments should have cascaded away"
    assert audit_row.model_name == "claude-x"
    assert audit_row.user_email == "alice@example.com"
    assert str(audit_row.model_id) == MODEL_ID
    assert str(audit_row.user_id) == USER_ID


def test_upgrade_downgrade_upgrade_roundtrip_on_clean_data(alembic_cfg, fresh_db_url):
    """On data where nothing was ever deleted, the migration must be fully reversible."""
    command.upgrade(alembic_cfg, "018")
    _seed_pre_019_data(fresh_db_url)
    command.upgrade(alembic_cfg, "019")

    command.downgrade(alembic_cfg, "018")
    command.upgrade(alembic_cfg, "head")

    async def _fetch(engine):
        async with engine.connect() as conn:
            return (await conn.execute(text("SELECT model_name, user_email FROM audit_logs"))).one()

    row = run_sql(fresh_db_url, _fetch)
    assert row.model_name == "claude-x"
    assert row.user_email == "alice@example.com"


def test_downgrade_refuses_after_hard_delete(alembic_cfg, fresh_db_url):
    """Once a model has actually been deleted, downgrading can't resurrect the FK
    constraint against data that now violates it -- and it shouldn't silently drop or
    rewrite that data to make the constraint fit. Assert this stays a hard failure,
    so a future "fix" doesn't quietly turn it into data loss."""
    command.upgrade(alembic_cfg, "018")
    _seed_pre_019_data(fresh_db_url)
    command.upgrade(alembic_cfg, "019")

    async def _delete_model(engine):
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM llm_models WHERE id = :id"), {"id": MODEL_ID})

    run_sql(fresh_db_url, _delete_model)

    with pytest.raises(IntegrityError):
        command.downgrade(alembic_cfg, "018")
