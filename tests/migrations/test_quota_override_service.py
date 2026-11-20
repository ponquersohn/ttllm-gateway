"""Quota override codes against a real Postgres: single use under concurrency, expiry,
revocation, and the per-request active-override lookup.

Plain ``def test_...`` driving one event loop through ``run_async`` (see conftest.py).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from alembic import command
from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tests.migrations.conftest import run_async
from ttllm.core.quota_override import OverrideState
from ttllm.models.quota import QuotaOverride
from ttllm.models.user import User
from ttllm.services import quota_override_service as svc
from ttllm.services.quota_override_service import InvalidOverrideCodeError

pytestmark = pytest.mark.migrations


@pytest.fixture
def migrated(alembic_cfg, fresh_db_url):
    command.upgrade(alembic_cfg, "head")
    return fresh_db_url


def _run(db_url, scenario):
    async def _go():
        engine = create_async_engine(db_url, pool_size=20)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            return await scenario(sessions)
        finally:
            await engine.dispose()

    return run_async(_go())


async def _user(sessions, name="u") -> User:
    async with sessions() as s:
        user = User(id=uuid.uuid4(), name=name, email=f"{name}-{uuid.uuid4().hex[:6]}@example.com")
        s.add(user)
        await s.commit()
        return user


async def _issue(sessions, issuer, override_seconds=3600, ttl=600, reason="testing"):
    async with sessions() as s:
        row, code = await svc.issue(s, issuer, override_seconds, ttl, reason)
        await s.commit()
        return row, code


def test_issue_stores_only_the_hash_and_previews(migrated):
    async def scenario(sessions):
        issuer = await _user(sessions, "issuer")
        row, code = await _issue(sessions, issuer)
        async with sessions() as s:
            preview = await svc.lookup(s, code)
            assert preview is not None and preview.id == row.id
            assert preview.code_hash != code and code not in preview.code_hash
            assert preview.reason == "testing" and preview.created_by_email == issuer.email
            assert await svc.lookup(s, code + "x") is None
            assert await svc.active_override_for_user(s, issuer.id) is None

    _run(migrated, scenario)


def test_redeem_is_single_use_even_concurrently(migrated):
    async def scenario(sessions):
        issuer = await _user(sessions, "issuer")
        users = [await _user(sessions, f"r{i}") for i in range(8)]
        _, code = await _issue(sessions, issuer, override_seconds=7200)

        async def attempt(user):
            async with sessions() as s:
                try:
                    row = await svc.redeem(s, code, user)
                    await s.commit()
                    return user, row
                except InvalidOverrideCodeError:
                    await s.rollback()
                    return user, None

        results = await asyncio.gather(*(attempt(u) for u in users))
        winners = [(u, r) for u, r in results if r is not None]
        assert len(winners) == 1
        winner, row = winners[0]
        assert row.redeemed_by == winner.id
        delta = (row.override_expires_at - row.redeemed_at).total_seconds()
        assert abs(delta - 7200) < 1

        async with sessions() as s:
            assert (await svc.active_override_for_user(s, winner.id)).id == row.id
            for u in users:
                if u.id != winner.id:
                    assert await svc.active_override_for_user(s, u.id) is None
            assert await svc.lookup(s, code) is None

    _run(migrated, scenario)


def test_issuer_can_redeem_own_code(migrated):
    async def scenario(sessions):
        issuer = await _user(sessions, "issuer")
        _, code = await _issue(sessions, issuer)
        async with sessions() as s:
            row = await svc.redeem(s, code, issuer)
            await s.commit()
            assert row.redeemed_by == issuer.id

    _run(migrated, scenario)


def test_expired_and_revoked_codes_cannot_be_redeemed(migrated):
    async def scenario(sessions):
        issuer = await _user(sessions, "issuer")
        user = await _user(sessions, "user")
        expired, expired_code = await _issue(sessions, issuer)
        revoked, revoked_code = await _issue(sessions, issuer)
        async with sessions() as s:
            await s.execute(
                update(QuotaOverride)
                .where(QuotaOverride.id == expired.id)
                .values(code_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
            )
            assert await svc.revoke(s, revoked.id, issuer) is not None
            await s.commit()
        for code in (expired_code, revoked_code):
            async with sessions() as s:
                with pytest.raises(InvalidOverrideCodeError):
                    await svc.redeem(s, code, user)
        async with sessions() as s:
            items, total = await svc.list_overrides(s, state=OverrideState.CODE_EXPIRED)
            assert total == 1 and items[0]["id"] == expired.id
            items, total = await svc.list_overrides(s, state=OverrideState.REVOKED)
            assert total == 1 and items[0]["id"] == revoked.id
            # A second revoke is a no-op.
            assert await svc.revoke(s, revoked.id, issuer) is None

    _run(migrated, scenario)


def test_revoking_or_expiry_ends_an_active_override(migrated):
    async def scenario(sessions):
        issuer = await _user(sessions, "issuer")
        user = await _user(sessions, "user")
        _, code_a = await _issue(sessions, issuer)
        _, code_b = await _issue(sessions, issuer)
        async with sessions() as s:
            a = await svc.redeem(s, code_a, user)
            await s.commit()
        async with sessions() as s:
            assert (await svc.active_override_for_user(s, user.id)).id == a.id
            items, total = await svc.list_overrides(s, state=OverrideState.ACTIVE)
            assert total == 1 and items[0]["redeemed_by_email"] == user.email
            await svc.revoke(s, a.id, issuer)
            await s.commit()
        async with sessions() as s:
            assert await svc.active_override_for_user(s, user.id) is None
            b = await svc.redeem(s, code_b, user)
            await s.commit()
        async with sessions() as s:
            await s.execute(
                update(QuotaOverride)
                .where(QuotaOverride.id == b.id)
                .values(override_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
            )
            await s.commit()
        async with sessions() as s:
            assert await svc.active_override_for_user(s, user.id) is None
            assert await svc.list_active_for_user(s, user.id) == []
            items, _ = await svc.list_overrides(s, state=OverrideState.EXPIRED)
            assert [i["id"] for i in items] == [b.id]

    _run(migrated, scenario)
