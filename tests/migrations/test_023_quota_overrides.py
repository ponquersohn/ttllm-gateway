"""Migration 023: quota_overrides table and the quota.override permission for administrators."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import text

from tests.migrations.conftest import run_sql

pytestmark = pytest.mark.migrations


def _state(db_url):
    async def _run(engine):
        async with engine.connect() as conn:
            tables = {r[0] for r in await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public'"))}
            perms = {
                r[0]
                for r in await conn.execute(
                    text(
                        "SELECT gp.permission FROM group_permissions gp JOIN groups g ON g.id = gp.group_id "
                        "WHERE g.name = 'administrators'"
                    )
                )
            }
            return tables, perms

    return run_sql(db_url, _run)


def test_upgrade_and_downgrade(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "023")
    tables, perms = _state(fresh_db_url)
    assert "quota_overrides" in tables
    assert "quota.override" in perms

    command.downgrade(alembic_cfg, "022")
    tables, perms = _state(fresh_db_url)
    assert "quota_overrides" not in tables
    assert "quota.override" not in perms
    assert "quota.view" in perms

    command.upgrade(alembic_cfg, "head")
