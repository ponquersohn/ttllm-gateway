"""Migration 020: quota tables, permissions, and disabling rules that used the old condition."""

from __future__ import annotations

import json
import uuid

import pytest
from alembic import command
from sqlalchemy import text

from tests.migrations.conftest import run_sql

pytestmark = pytest.mark.migrations

_QUOTA = {"type": "quota", "field": "cost", "operator": "gt", "value": 5, "window": 60}
_PARAM = {"type": "parameter", "field": "model", "operator": "exact", "value": "m"}
_BLOCK = {"type": "block", "status_code": 429}


def _seed_rule(db_url, name, conditions, action=_BLOCK, description=None, enabled=True):
    async def _run(engine):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO rules (id, name, description, weight, enabled, conditions, action)
                    VALUES (:id, :name, :description, 0, :enabled, CAST(:conditions AS jsonb), CAST(:action AS jsonb))
                    """
                ),
                {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "description": description,
                    "enabled": enabled,
                    "conditions": json.dumps(conditions),
                    "action": json.dumps(action),
                },
            )

    run_sql(db_url, _run)


def _rule(db_url, name):
    async def _run(engine):
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT enabled, description FROM rules WHERE name = :name"), {"name": name}
            )
            return result.one()

    return run_sql(db_url, _run)


def _tables(db_url):
    async def _run(engine):
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'"))
            return {row[0] for row in result}

    return run_sql(db_url, _run)


def test_disables_rules_with_quota_conditions_at_any_depth(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "019")
    _seed_rule(fresh_db_url, "top-level", {"logic": "and", "conditions": [_QUOTA]}, description="daily cap")
    _seed_rule(
        fresh_db_url,
        "nested",
        {"logic": "and", "conditions": [_PARAM, {"logic": "or", "conditions": [_PARAM, _QUOTA]}]},
    )
    _seed_rule(fresh_db_url, "unrelated", {"logic": "and", "conditions": [_PARAM]}, description="keep me")
    _seed_rule(fresh_db_url, "long", {"logic": "and", "conditions": [_QUOTA]}, description="x" * 1024)

    command.upgrade(alembic_cfg, "020")

    enabled, description = _rule(fresh_db_url, "top-level")
    assert enabled is False
    assert "migration 020" in description and "daily cap" in description

    assert _rule(fresh_db_url, "nested").enabled is False

    enabled, description = _rule(fresh_db_url, "unrelated")
    assert enabled is True
    assert description == "keep me"

    enabled, description = _rule(fresh_db_url, "long")
    assert enabled is False
    assert len(description) <= 1024


def test_grants_quota_permissions_to_administrators(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "020")

    async def _run(engine):
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT gp.permission FROM group_permissions gp
                    JOIN groups g ON g.id = gp.group_id
                    WHERE g.name = 'administrators' AND gp.permission LIKE 'quota.%'
                    """
                )
            )
            return {row[0] for row in result}

    assert run_sql(fresh_db_url, _run) == {"quota.view", "quota.create", "quota.modify", "quota.delete"}


def test_round_trip(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "020")
    assert {"quotas", "quota_buckets", "quota_reservations"} <= _tables(fresh_db_url)

    _seed_rule(
        fresh_db_url,
        "attaches-quota",
        {"logic": "and", "conditions": []},
        action={"type": "apply_quota", "quota_ids": [str(uuid.uuid4())]},
    )
    _seed_rule(
        fresh_db_url,
        "uses-subject",
        {"logic": "and", "conditions": [{"type": "subject", "field": "group", "value": "eng"}]},
    )
    _seed_rule(fresh_db_url, "plain", {"logic": "and", "conditions": [_PARAM]})

    command.downgrade(alembic_cfg, "019")
    assert not {"quotas", "quota_buckets", "quota_reservations"} & _tables(fresh_db_url)
    assert _rule(fresh_db_url, "attaches-quota").enabled is False
    assert _rule(fresh_db_url, "uses-subject").enabled is False
    assert _rule(fresh_db_url, "plain").enabled is True

    command.upgrade(alembic_cfg, "020")
    assert {"quotas", "quota_buckets", "quota_reservations"} <= _tables(fresh_db_url)
