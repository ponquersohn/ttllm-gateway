"""Generic migration-hygiene checks, hand-rolled from Alembic's own public API rather
than pulling in pytest-alembic (whose sync-engine convenience layer doesn't sit
naturally on our asyncpg-only, settings-driven env.py -- see conftest.py)."""

from __future__ import annotations

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

from tests.migrations.conftest import run_async
from ttllm.models import Base

pytestmark = pytest.mark.migrations


def test_single_head_revision(alembic_cfg):
    """Catches an accidental branch split (two migrations both claiming the same
    down_revision), which alembic upgrade head would otherwise apply nondeterministically."""
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"expected exactly one head revision, got {heads}"


def test_orm_matches_migrations(alembic_cfg, engine):
    """After `upgrade head`, the live schema must match Base.metadata exactly.

    Catches the easy-to-make mistake in this codebase's workflow: editing a SQLAlchemy
    model (models/*.py) without writing (or correctly writing) the matching migration --
    autogenerate.compare_metadata reports the actual DDL diff, not just "some difference".
    """
    command.upgrade(alembic_cfg, "head")

    def _compare(sync_conn):
        context = MigrationContext.configure(sync_conn)
        return compare_metadata(context, Base.metadata)

    async def _reflect_and_compare():
        async with engine.connect() as conn:
            return await conn.run_sync(_compare)

    diffs = run_async(_reflect_and_compare())
    assert diffs == [], f"ORM/migration drift detected: {diffs}"
