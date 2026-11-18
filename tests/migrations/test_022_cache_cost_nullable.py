"""Migration 022: cache prices become nullable, and the old default 0 becomes NULL
(bill at the input rate) while real prices are kept."""

from __future__ import annotations

import pytest
from alembic import command
from sqlalchemy import text

from tests.migrations.conftest import run_sql

pytestmark = pytest.mark.migrations

UNSET_ID = "11111111-1111-1111-1111-111111111111"
PRICED_ID = "22222222-2222-2222-2222-222222222222"


def _seed(db_url: str) -> None:
    async def _run(engine):
        async with engine.begin() as conn:
            for model_id, name, read, write in (
                (UNSET_ID, "unset", "0", "0"),
                (PRICED_ID, "priced", "0.0003", "0.00375"),
            ):
                await conn.execute(
                    text(
                        """
                        INSERT INTO llm_models (id, name, provider, provider_model_id, config_json, is_active,
                                                input_cost_per_1k, output_cost_per_1k,
                                                cache_read_cost_per_1k, cache_write_cost_per_1k)
                        VALUES (:id, :name, 'bedrock', 'm', '{}'::jsonb, true, 0.003, 0.015, :read, :write)
                        """
                    ),
                    {"id": model_id, "name": name, "read": read, "write": write},
                )

    run_sql(db_url, _run)


def _prices(db_url: str) -> dict[str, tuple]:
    async def _run(engine):
        async with engine.connect() as conn:
            rows = await conn.execute(
                text("SELECT name, cache_read_cost_per_1k, cache_write_cost_per_1k FROM llm_models")
            )
            return {r.name: (r.cache_read_cost_per_1k, r.cache_write_cost_per_1k) for r in rows}

    return run_sql(db_url, _run)


def test_default_zero_becomes_null_and_prices_are_kept(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "021")
    _seed(fresh_db_url)
    command.upgrade(alembic_cfg, "022")

    prices = _prices(fresh_db_url)
    assert prices["unset"] == (None, None)
    assert [str(p) for p in prices["priced"]] == ["0.000300", "0.003750"]


def test_downgrade_restores_zero(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "021")
    _seed(fresh_db_url)
    command.upgrade(alembic_cfg, "022")
    command.downgrade(alembic_cfg, "021")

    prices = _prices(fresh_db_url)
    assert [str(p) for p in prices["unset"]] == ["0.000000", "0.000000"]
    command.upgrade(alembic_cfg, "head")
