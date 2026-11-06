"""Quota reserve/settle against a real Postgres.

Row locks, ON CONFLICT and rollback behaviour are the whole point of this code, and mocks
can't stand in for them. These live here for the disposable-Postgres fixtures, and follow
that directory's rule: plain ``def test_...`` driving one event loop through ``run_async``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from alembic import command
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from tests.migrations.conftest import run_async
from ttllm.config import settings
from ttllm.core.quota import ActualUsage, QuotaSubject, RequestEstimate
from ttllm.models.quota import Quota, QuotaBucket, QuotaReservation
from ttllm.services import quota_service
from ttllm.services.quota_service import QuotaContentionError, QuotaExceededError

pytestmark = pytest.mark.migrations

ESTIMATE = RequestEstimate(input_chars=400, max_tokens=100)  # 200 tokens


def _subject(user: str = "u1", model: str = "m1", token: str = "t1") -> QuotaSubject:
    return QuotaSubject(user, f"{user}@example.com", model, f"name-{model}", token)


class _Harness:
    """A session factory bound to this test's database, swapped into the service."""

    def __init__(self, db_url: str, monkeypatch):
        self.engine = create_async_engine(db_url, pool_size=30, max_overflow=30)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        monkeypatch.setattr(quota_service, "async_session_factory", self.sessions)

    async def add_quota(self, **overrides) -> str:
        fields = dict(
            id=uuid.uuid4(),
            name=f"q-{uuid.uuid4().hex[:8]}",
            measure="requests",
            limit_value=Decimal(5),
            dimensions=["user"],
            window_type="calendar",
            window_unit="day",
            window_duration_seconds=None,
            timezone="UTC",
            enforcement="soft",
            on_exceeded={},
            enabled=True,
        )
        fields.update(overrides)
        async with self.sessions() as session:
            session.add(Quota(**fields))
            await session.commit()
        return str(fields["id"])

    async def settle(self, lease, usage: ActualUsage) -> None:
        async with self.sessions() as session:
            await quota_service.settle(session, lease, usage)
            await session.commit()

    async def buckets(self, quota_id: str) -> list:
        async with self.sessions() as session:
            result = await session.execute(select(QuotaBucket).where(QuotaBucket.quota_id == uuid.UUID(quota_id)))
            return list(result.scalars().all())

    async def reservation_count(self) -> int:
        async with self.sessions() as session:
            return len((await session.execute(select(QuotaReservation.id))).all())

    async def expire_window(self, quota_id: str) -> None:
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        async with self.sessions() as session:
            await session.execute(
                update(QuotaBucket)
                .where(QuotaBucket.quota_id == uuid.UUID(quota_id))
                .values(window_start=past - timedelta(hours=1), window_end=past)
            )
            await session.commit()

    async def expire_reservations(self, older_by: timedelta = timedelta(seconds=1)) -> None:
        async with self.sessions() as session:
            await session.execute(
                update(QuotaReservation).values(expires_at=datetime.now(timezone.utc) - older_by)
            )
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()


@pytest.fixture
def harness(alembic_cfg, fresh_db_url, monkeypatch):
    command.upgrade(alembic_cfg, "head")
    return _Harness(fresh_db_url, monkeypatch)


def _run(harness: _Harness, scenario):
    async def _main():
        try:
            return await scenario()
        finally:
            await harness.close()

    return run_async(_main())


async def _reserve(quota_ids, subject=None, estimate=ESTIMATE):
    return await quota_service.reserve(quota_ids, subject or _subject(), estimate, uuid.uuid4())


def test_parallel_requests_cannot_exceed_a_request_limit(harness):
    """The reason reservations exist: with only a lock, all of these would be admitted."""

    async def scenario():
        quota = await harness.add_quota(limit_value=Decimal(5))
        results = await asyncio.gather(*[_reserve([quota]) for _ in range(25)], return_exceptions=True)
        admitted = [r for r in results if isinstance(r, quota_service.QuotaLease)]
        denied = [r for r in results if isinstance(r, QuotaExceededError)]
        assert len(admitted) == 5
        assert len(denied) == 20
        assert len(await harness.buckets(quota)) == 1

    _run(harness, scenario)


def test_soft_token_quota_overshoots_by_about_one_request(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(100))
        results = await asyncio.gather(*[_reserve([quota]) for _ in range(10)], return_exceptions=True)
        # The first reserves 200 tokens against a limit of 100, and everyone else sees that.
        assert sum(isinstance(r, quota_service.QuotaLease) for r in results) == 1

    _run(harness, scenario)


def test_strict_estimate_larger_than_the_limit_has_no_retry_after(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(100), enforcement="strict")
        with pytest.raises(QuotaExceededError) as exc:
            await _reserve([quota])
        assert exc.value.denials[0].retry_after is None

    _run(harness, scenario)


def test_overlapping_quotas_in_opposite_order_do_not_deadlock(harness):
    async def scenario():
        a = await harness.add_quota(limit_value=Decimal(1000))
        b = await harness.add_quota(limit_value=Decimal(1000))
        calls = [_reserve([a, b] if i % 2 else [b, a]) for i in range(60)]
        results = await asyncio.gather(*calls, return_exceptions=True)
        assert [r for r in results if isinstance(r, Exception)] == []

    _run(harness, scenario)


def test_two_windows_of_the_same_measure_are_independent(harness):
    """Regression: the old engine kept one value per measure, so the documented
    $50/day + $20/hour pair only ever compared against whichever window ran last."""

    async def scenario():
        daily = await harness.add_quota(name="daily", measure="cost", limit_value=Decimal(50), window_unit="day")
        hourly = await harness.add_quota(name="hourly", measure="cost", limit_value=Decimal(20), window_unit="hour")

        lease = await _reserve([daily, hourly])
        await harness.settle(lease, ActualUsage(200, tokens=1, cost=Decimal(15)))

        # A new hour: the hourly window clears, the daily one keeps its spend.
        await harness.expire_window(hourly)
        lease = await _reserve([daily, hourly])
        await harness.settle(lease, ActualUsage(200, tokens=1, cost=Decimal(15)))
        await harness.expire_window(hourly)
        lease = await _reserve([daily, hourly])
        await harness.settle(lease, ActualUsage(200, tokens=1, cost=Decimal(15)))
        await harness.expire_window(hourly)
        lease = await _reserve([daily, hourly])
        await harness.settle(lease, ActualUsage(200, tokens=1, cost=Decimal(15)))

        # $60 today but only $15 this hour: the daily cap must be the one that blocks.
        await harness.expire_window(hourly)
        with pytest.raises(QuotaExceededError) as exc:
            await _reserve([daily, hourly])
        assert [d.name for d in exc.value.denials] == ["daily"]

    _run(harness, scenario)


def test_denial_on_one_quota_reserves_nothing_on_the_others(harness):
    async def scenario():
        roomy = await harness.add_quota(limit_value=Decimal(100))
        full = await harness.add_quota(limit_value=Decimal(1))
        first = await _reserve([full])
        count = await harness.reservation_count()
        with pytest.raises(QuotaExceededError):
            await _reserve([roomy, full])
        assert await harness.reservation_count() == count
        await harness.settle(first, ActualUsage(200, tokens=1))

    _run(harness, scenario)


def test_settle_swaps_the_estimate_for_actual_usage(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(1000))
        lease = await _reserve([quota])
        assert await harness.reservation_count() == 1
        await harness.settle(lease, ActualUsage(200, tokens=37))
        assert await harness.reservation_count() == 0
        assert (await harness.buckets(quota))[0].used == 37

    _run(harness, scenario)


def test_settling_twice_only_charges_once(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(1000))
        lease = await _reserve([quota])
        await harness.settle(lease, ActualUsage(200, tokens=37))
        await harness.settle(lease, ActualUsage(200, tokens=37))
        assert (await harness.buckets(quota))[0].used == 37

    _run(harness, scenario)


def test_provider_error_refunds_the_slot(harness):
    async def scenario():
        quota = await harness.add_quota(limit_value=Decimal(1))
        lease = await _reserve([quota])
        await harness.settle(lease, ActualUsage(500))
        assert (await harness.buckets(quota))[0].used == 0
        await _reserve([quota])

    _run(harness, scenario)


def test_parallel_settles_do_not_lose_updates(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(10_000_000))
        leases = await asyncio.gather(*[_reserve([quota]) for _ in range(20)])
        await asyncio.gather(*[harness.settle(lease, ActualUsage(200, tokens=7)) for lease in leases])
        assert (await harness.buckets(quota))[0].used == 140

    _run(harness, scenario)


def test_crashed_workers_reservation_stops_counting_after_ttl(harness):
    async def scenario():
        quota = await harness.add_quota(limit_value=Decimal(1))
        await _reserve([quota])  # never settled
        with pytest.raises(QuotaExceededError):
            await _reserve([quota])
        await harness.expire_reservations()
        await _reserve([quota])

    _run(harness, scenario)


def test_stream_outliving_the_ttl_is_still_charged(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(1000))
        lease = await _reserve([quota])
        await harness.expire_reservations()  # past the TTL, within retention
        await _reserve([quota], _subject("other-user"))  # opportunistic cleanup runs
        await _reserve([quota])  # including on this very bucket
        await harness.settle(lease, ActualUsage(200, tokens=37))
        buckets = await harness.buckets(quota)
        assert sum(b.used for b in buckets) == 37

    _run(harness, scenario)


def test_reservations_past_retention_are_cleaned_up(harness):
    async def scenario():
        quota = await harness.add_quota(limit_value=Decimal(100))
        await _reserve([quota])
        retention = settings.engine.quota_reservation_retention_seconds
        await harness.expire_reservations(older_by=timedelta(seconds=retention + 60))
        await _reserve([quota])
        assert await harness.reservation_count() == 1

    _run(harness, scenario)


def test_expired_window_rolls_over_on_the_next_request(harness):
    async def scenario():
        quota = await harness.add_quota(limit_value=Decimal(1))
        lease = await _reserve([quota])
        await harness.settle(lease, ActualUsage(200, tokens=1))
        with pytest.raises(QuotaExceededError):
            await _reserve([quota])
        await harness.expire_window(quota)
        await _reserve([quota])
        bucket = (await harness.buckets(quota))[0]
        assert bucket.used == 0
        assert bucket.window_end > datetime.now(timezone.utc)

    _run(harness, scenario)


def test_session_window_starts_at_the_first_request(harness):
    async def scenario():
        quota = await harness.add_quota(window_type="session", window_unit=None, window_duration_seconds=10800)
        before = datetime.now(timezone.utc)
        await _reserve([quota])
        bucket = (await harness.buckets(quota))[0]
        assert abs(bucket.window_start - before) < timedelta(seconds=5)
        assert bucket.window_end - bucket.window_start == timedelta(hours=3)

    _run(harness, scenario)


def test_settling_after_a_session_expired_does_not_start_a_new_session(harness):
    """A window with no request behind it would make the next user's window start early."""

    async def scenario():
        quota = await harness.add_quota(
            measure="tokens",
            limit_value=Decimal(1000),
            window_type="session",
            window_unit=None,
            window_duration_seconds=60,
        )
        lease = await _reserve([quota])
        await harness.expire_window(quota)
        expired_end = (await harness.buckets(quota))[0].window_end
        await harness.settle(lease, ActualUsage(200, tokens=5))
        assert (await harness.buckets(quota))[0].window_end == expired_end

    _run(harness, scenario)


def test_reservations_from_a_previous_window_do_not_count(harness):
    async def scenario():
        quota = await harness.add_quota(
            limit_value=Decimal(1), window_type="session", window_unit=None, window_duration_seconds=60
        )
        await _reserve([quota])  # never settled, and its TTL hasn't passed
        await harness.expire_window(quota)
        await _reserve([quota])

    _run(harness, scenario)


def test_new_limit_applies_immediately_and_keeps_usage(harness):
    async def scenario():
        quota_id = await harness.add_quota(limit_value=Decimal(1))
        lease = await _reserve([quota_id])
        await harness.settle(lease, ActualUsage(200, tokens=1))
        with pytest.raises(QuotaExceededError):
            await _reserve([quota_id])

        async with harness.sessions() as session:
            quota = await session.get(Quota, uuid.UUID(quota_id))
            await quota_service.update_quota(session, quota, {"limit_value": Decimal(2)})

        assert (await harness.buckets(quota_id))[0].used == 1
        await _reserve([quota_id])

    _run(harness, scenario)


def test_changing_the_window_discards_buckets(harness):
    async def scenario():
        quota_id = await harness.add_quota()
        await _reserve([quota_id])

        async with harness.sessions() as session:
            quota = await session.get(Quota, uuid.UUID(quota_id))
            await quota_service.update_quota(session, quota, {"window_unit": "hour"})

        assert await harness.buckets(quota_id) == []

    _run(harness, scenario)


def test_settling_against_a_reset_quota_is_a_no_op(harness):
    async def scenario():
        quota = await harness.add_quota(measure="tokens", limit_value=Decimal(1000))
        lease = await _reserve([quota])
        async with harness.sessions() as session:
            assert await quota_service.reset_quota(session, uuid.UUID(quota)) == 1
        await harness.settle(lease, ActualUsage(200, tokens=5))
        assert await harness.buckets(quota) == []

    _run(harness, scenario)


def test_dimensions_separate_buckets(harness):
    async def scenario():
        per_user = await harness.add_quota(limit_value=Decimal(1), dimensions=["user"])
        await _reserve([per_user], _subject("alice"))
        await _reserve([per_user], _subject("bob"))
        with pytest.raises(QuotaExceededError):
            await _reserve([per_user], _subject("alice"))

        shared = await harness.add_quota(limit_value=Decimal(1), dimensions=[])
        await _reserve([shared], _subject("alice"))
        with pytest.raises(QuotaExceededError):
            await _reserve([shared], _subject("bob"))

        per_model = await harness.add_quota(limit_value=Decimal(1), dimensions=["user", "model"])
        await _reserve([per_model], _subject("alice", "m1"))
        await _reserve([per_model], _subject("alice", "m2"))

    _run(harness, scenario)


def test_disabled_and_deleted_quotas_are_skipped(harness):
    async def scenario():
        disabled = await harness.add_quota(limit_value=Decimal(1), enabled=False)
        for _ in range(3):
            lease = await _reserve([disabled, str(uuid.uuid4())])
            assert lease.reservations == ()

    _run(harness, scenario)


def test_lock_timeout_surfaces_as_contention(harness, monkeypatch):
    async def scenario():
        monkeypatch.setattr(settings.engine, "quota_lock_timeout_ms", 50)
        quota = await harness.add_quota()
        await _reserve([quota])
        bucket_id = (await harness.buckets(quota))[0].id

        async with harness.sessions() as holder, holder.begin():
            await holder.execute(select(QuotaBucket.id).where(QuotaBucket.id == bucket_id).with_for_update())
            with pytest.raises(QuotaContentionError):
                await _reserve([quota])

    _run(harness, scenario)


def test_headers_report_the_tightest_bucket(harness):
    async def scenario():
        roomy = await harness.add_quota(name="roomy", limit_value=Decimal(100))
        tight = await harness.add_quota(name="tight", limit_value=Decimal(2))
        lease = await _reserve([roomy, tight])
        assert lease.headers["x-ttllm-quota-name"] == "tight"
        assert lease.headers["x-ttllm-quota-limit"] == "2"
        assert lease.headers["x-ttllm-quota-remaining"] == "1"
        assert int(lease.headers["x-ttllm-quota-reset"]) > 0

    _run(harness, scenario)
