"""Fixtures for testing Alembic migrations against a real, disposable Postgres.

These tests need a real Postgres (JSONB round-tripping, ON DELETE CASCADE alteration,
partial/composite indexes -- none of which SQLite can stand in for), spun up via
testcontainers rather than depending on docker-compose already being up. Assertions go
through the app's actual asyncpg driver (not a sync driver) so driver-specific
serialization bugs -- like the JSONB double-encoding bug caught while writing migration
019 -- would actually be caught here.

Key trick: ``alembic/env.py`` always reads ``ttllm.config.settings.database.url`` at
call time (it ignores whatever's in alembic.ini). ``ttllm.config`` is a process-wide
singleton module, so the only reliable way to redirect a migration at a disposable test
database -- independent of whatever TTLLM_CONFIG_FILE/env vars happen to be set in the
test process, and regardless of import order -- is to mutate that singleton in place via
monkeypatch, not environment variables.

Constraint: alembic.command.upgrade/downgrade call asyncio.run() internally (via env.py).
That collides with pytest-asyncio's already-running loop, so every test under this
directory must be a plain `def test_...`, not `async def`. Use `run_async()` below for
any async DB call a test needs to make directly.

IMPORTANT -- always run this directory as its own process (`pytest -m migrations`, the
default `pytest` excludes it via addopts): `alembic.config.Config` internally calls
`logging.config.fileConfig()`, which reconfigures Python's *global* logging setup. Run
these in the same process as the rest of the suite (e.g. `pytest -m "not integration"`,
which does NOT exclude `migrations`) and it silently corrupts unrelated tests' `caplog`
expectations later in that same run. This bit us once already -- don't remove the
`migrations` exclusion from `addopts` to "simplify" the invocation.
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

import pytest
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ttllm.config import settings

with warnings.catch_warnings():
    # testcontainers.postgres (the module path used by older docs/examples) is
    # deprecated in favor of testcontainers.community.postgres; import the
    # maintained path directly so this doesn't print a warning on every run.
    warnings.simplefilter("ignore", DeprecationWarning)
    from testcontainers.community.postgres import PostgresContainer

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
ALEMBIC_DIR = REPO_ROOT / "alembic"

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from a plain sync test function (see module docstring)."""
    return asyncio.run(coro)


def run_sql(db_url: str, fn: Callable[[AsyncEngine], Coroutine[Any, Any, T]]) -> T:
    """Run `fn(engine)` against a fresh, short-lived engine bound to `db_url`, created
    and disposed inside its own event loop.

    Use this -- not the shared `engine` fixture -- for any test that needs more than one
    `run_async()` call (e.g. seed, then migrate, then query): an AsyncEngine's connection
    pool binds to whichever event loop first uses it, and `run_async()`/`asyncio.run()`
    spins a brand-new loop every call, so reusing one engine across calls fails with
    "Future attached to a different loop". The `engine` fixture is only safe for a test
    that makes exactly one `run_async()` call.
    """

    async def _run() -> T:
        eng = create_async_engine(db_url)
        try:
            return await fn(eng)
        finally:
            await eng.dispose()

    return run_async(_run())


@pytest.fixture(scope="session")
def pg_container():
    """One Postgres container for the whole session -- tests get isolation via a fresh
    database each (see `fresh_db_url`), not a fresh container, since container startup
    is the dominant cost."""
    with PostgresContainer("postgres:16", driver="asyncpg") as container:
        yield container


@pytest.fixture
def fresh_db_url(pg_container) -> str:
    """A brand-new, empty database inside the shared container, for this test only."""
    admin_url = pg_container.get_connection_url()
    db_name = f"test_{uuid.uuid4().hex}"

    async def _create() -> None:
        engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        finally:
            await engine.dispose()

    run_async(_create())

    yield admin_url.rsplit("/", 1)[0] + f"/{db_name}"

    async def _drop() -> None:
        engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                await conn.execute(text(f'DROP DATABASE "{db_name}" WITH (FORCE)'))
        finally:
            await engine.dispose()

    run_async(_drop())


@pytest.fixture
def alembic_cfg(fresh_db_url, monkeypatch) -> Config:
    """An Alembic Config wired at `fresh_db_url` via the settings-singleton trick above."""
    monkeypatch.setattr(settings.database, "url", fresh_db_url)
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    return cfg


@pytest.fixture
def engine(fresh_db_url) -> AsyncEngine:
    """Convenience for a test that makes exactly one `run_async()` call. For anything
    that seeds/migrates/queries in separate steps, use `run_sql(fresh_db_url, ...)`."""
    eng = create_async_engine(fresh_db_url)
    yield eng
    run_async(eng.dispose())
