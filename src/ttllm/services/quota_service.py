"""Quota definitions, and the reserve/settle transactions that enforce them.

Enforcement has two short transactions around the provider call:

* ``reserve`` locks the request's buckets, rolls expired windows, checks limits, and records
  the request's estimate as a reservation. It commits before the provider is called, so no
  lock or connection is held during the call.
* ``settle`` swaps the reservation for actual usage, in the caller's audit transaction.

Reservations are what make parallel requests safe. The row lock only serializes the check,
and ``used`` doesn't change until a response finishes, so without them every parallel
request would see the same remaining budget.

``reserve`` opens its own session rather than using the request's: a denial rolls back, and
rolling back the request session would expire the ORM objects the handler still needs.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.core.quota import (
    ActualUsage,
    QuotaDenial,
    QuotaSubject,
    RequestEstimate,
    WindowSpec,
    WindowType,
    WindowUnit,
    actual_amount,
    admits,
    bucket_identity,
    compute_window,
    estimate_amount,
    format_amount,
    seconds_until,
)
from ttllm.core.rules import collect_quota_ids
from ttllm.db import async_session_factory
from ttllm.models.quota import Quota, QuotaBucket, QuotaReservation
from ttllm.models.rule import Rule
from ttllm.schemas.anthropic import MessagesRequest
from ttllm.services import model_service, rules_service

logger = logging.getLogger(__name__)

_quotas = Quota.__table__
_buckets = QuotaBucket.__table__
_reservations = QuotaReservation.__table__

# lock_not_available (lock_timeout expired) and deadlock_detected.
_CONTENTION_SQLSTATES = {"55P03", "40P01"}

# Changing any of these changes what a bucket means, so its counters can't carry over.
_BUCKET_DEFINING_FIELDS = {
    "measure",
    "dimensions",
    "window_type",
    "window_unit",
    "window_duration_seconds",
    "timezone",
}


class QuotaExceededError(Exception):
    def __init__(self, denials: list[QuotaDenial]):
        super().__init__(", ".join(d.name for d in denials))
        self.denials = denials


class QuotaContentionError(Exception):
    """Couldn't get bucket locks in time. Retryable, so it surfaces as a 529."""


class QuotaInUseError(Exception):
    """The quota is still attached to rules."""


@dataclass(frozen=True)
class _Reservation:
    reservation_id: uuid.UUID
    bucket_id: uuid.UUID
    measure: str
    amount: Decimal


@dataclass(frozen=True)
class QuotaLease:
    """What ``reserve`` hands back. Reservations are in lock order, which settle relies on."""

    reservations: tuple[_Reservation, ...] = ()
    headers: dict[str, str] = field(default_factory=dict)


NO_LEASE = QuotaLease()


def _window_spec(quota: Any) -> WindowSpec:
    return WindowSpec(
        type=WindowType(quota.window_type),
        unit=WindowUnit(quota.window_unit) if quota.window_unit else None,
        duration_seconds=quota.window_duration_seconds,
        timezone=quota.timezone,
    )


def _is_expired(window_end: datetime | None, now: datetime) -> bool:
    return window_end is None or window_end <= now


async def _set_lock_timeout(session: AsyncSession) -> None:
    # SET LOCAL can't take bind parameters; set_config(..., is_local => true) is equivalent.
    await session.execute(
        select(func.set_config("lock_timeout", str(settings.engine.quota_lock_timeout_ms), True))
    )


async def _in_flight_by_bucket(
    session: AsyncSession,
    windows: dict[uuid.UUID, datetime | None],
    now: datetime,
) -> dict[uuid.UUID, Decimal]:
    """Unexpired reservations against each bucket's *current* window."""
    if not windows:
        return {}
    rows = await session.execute(
        select(
            _reservations.c.bucket_id,
            _reservations.c.window_start,
            func.sum(_reservations.c.amount).label("amount"),
        )
        .where(_reservations.c.bucket_id.in_(list(windows)), _reservations.c.expires_at > now)
        .group_by(_reservations.c.bucket_id, _reservations.c.window_start)
    )
    in_flight: dict[uuid.UUID, Decimal] = {}
    for row in rows:
        if windows[row.bucket_id] == row.window_start:
            in_flight[row.bucket_id] = row.amount
    return in_flight


async def reserve(
    quota_ids: list[str],
    subject: QuotaSubject,
    estimate: RequestEstimate,
    request_id: uuid.UUID,
) -> QuotaLease:
    """Admit a request against all its quotas, or raise ``QuotaExceededError``.

    All-or-nothing: a denial on any bucket rolls back every reservation. Buckets are created
    and locked in ``(quota_id, bucket_key)`` order, so overlapping requests can't deadlock.
    """
    if not quota_ids:
        return NO_LEASE
    try:
        async with async_session_factory() as session, session.begin():
            return await _reserve(session, quota_ids, subject, estimate, request_id)
    except DBAPIError as exc:
        sqlstate = getattr(exc.orig, "sqlstate", None) or getattr(exc.orig, "pgcode", None)
        if sqlstate in _CONTENTION_SQLSTATES:
            raise QuotaContentionError(str(exc)) from exc
        raise


async def _reserve(
    session: AsyncSession,
    quota_ids: list[str],
    subject: QuotaSubject,
    estimate: RequestEstimate,
    request_id: uuid.UUID,
) -> QuotaLease:
    requested = {uuid.UUID(q) for q in quota_ids}
    rows = (
        await session.execute(select(_quotas).where(_quotas.c.id.in_(list(requested)), _quotas.c.enabled.is_(True)))
    ).all()
    quotas = {row.id: row for row in rows}
    # A worker's rule cache can briefly outlive a quota being disabled or deleted.
    stale = requested - set(quotas)
    if stale:
        logger.warning("Skipping unknown or disabled quotas: %s", sorted(str(q) for q in stale))
    if not quotas:
        return NO_LEASE

    identities = {
        (quota.id, key): values
        for quota in quotas.values()
        for key, values in [bucket_identity(tuple(quota.dimensions), subject)]
    }
    ordered = sorted(identities, key=lambda pair: (str(pair[0]), pair[1]))

    await _set_lock_timeout(session)
    await session.execute(
        pg_insert(_buckets)
        .values(
            [
                {
                    "id": uuid.uuid4(),
                    "quota_id": quota_id,
                    "bucket_key": key,
                    "dimension_values": identities[(quota_id, key)],
                    "used": Decimal(0),
                }
                for quota_id, key in ordered
            ]
        )
        .on_conflict_do_nothing(index_elements=["quota_id", "bucket_key"])
    )
    locked = (
        await session.execute(
            select(_buckets)
            .where(
                _buckets.c.quota_id.in_(list(quotas)),
                _buckets.c.bucket_key.in_([key for _, key in ordered]),
            )
            .order_by(_buckets.c.quota_id, _buckets.c.bucket_key)
            .with_for_update()
        )
    ).all()
    # Read the clock after the locks: time spent queueing must not count as "before now".
    now: datetime = (await session.execute(select(func.clock_timestamp()))).scalar_one()

    # (bucket row, its quota, window_start, window_end, used), after lazy rollover.
    current = []
    for bucket in locked:
        if (bucket.quota_id, bucket.bucket_key) not in identities:
            continue
        quota = quotas[bucket.quota_id]
        window_start, window_end, used = bucket.window_start, bucket.window_end, bucket.used
        if _is_expired(window_end, now):
            window_start, window_end = compute_window(_window_spec(quota), now)
            used = Decimal(0)
            await session.execute(
                update(_buckets)
                .where(_buckets.c.id == bucket.id)
                .values(window_start=window_start, window_end=window_end, used=0, updated_at=now)
            )
        current.append((bucket, quota, window_start, window_end, used))

    retention = timedelta(seconds=settings.engine.quota_reservation_retention_seconds)
    bucket_ids = [entry[0].id for entry in current]
    await session.execute(
        delete(_reservations).where(
            _reservations.c.bucket_id.in_(bucket_ids),
            _reservations.c.expires_at < now - retention,
        )
    )
    in_flight = await _in_flight_by_bucket(session, {b.id: start for b, _, start, _, _ in current}, now)

    denials: list[QuotaDenial] = []
    leases: list[_Reservation] = []
    expires_at = now + timedelta(seconds=settings.engine.quota_reservation_ttl_seconds)
    reservation_rows: list[dict[str, Any]] = []
    tightest: tuple[Decimal, dict[str, str]] | None = None

    for bucket, quota, window_start, window_end, used in current:
        amount = estimate_amount(quota.measure, estimate)
        bucket_in_flight = in_flight.get(bucket.id, Decimal(0))
        limit = quota.limit_value
        if not admits(quota.enforcement, used, bucket_in_flight, amount, limit):
            # Only an estimate larger than the whole limit can't be helped by waiting.
            never_fits = quota.enforcement == "strict" and amount > limit
            denials.append(
                QuotaDenial(
                    quota_id=str(quota.id),
                    name=quota.name,
                    measure=quota.measure,
                    limit=limit,
                    used=used,
                    in_flight=bucket_in_flight,
                    retry_after=None if never_fits else seconds_until(window_end, now),
                    on_exceeded=quota.on_exceeded or {},
                )
            )
            continue

        reservation_id = uuid.uuid4()
        reservation_rows.append(
            {
                "id": reservation_id,
                "bucket_id": bucket.id,
                "request_id": request_id,
                "amount": amount,
                "window_start": window_start,
                "expires_at": expires_at,
            }
        )
        leases.append(_Reservation(reservation_id, bucket.id, quota.measure, amount))

        remaining = max(limit - used - bucket_in_flight - amount, Decimal(0))
        fraction_left = remaining / limit
        if tightest is None or fraction_left < tightest[0]:
            tightest = (
                fraction_left,
                {
                    "x-ttllm-quota-name": quota.name,
                    "x-ttllm-quota-limit": format_amount(limit),
                    "x-ttllm-quota-remaining": format_amount(remaining),
                    "x-ttllm-quota-reset": str(seconds_until(window_end, now)),
                },
            )

    if denials:
        raise QuotaExceededError(denials)

    await session.execute(_reservations.insert(), reservation_rows)
    return QuotaLease(reservations=tuple(leases), headers=tightest[1] if tightest else {})


async def settle(session: AsyncSession, lease: QuotaLease, usage: ActualUsage) -> None:
    """Replace a lease's reservations with actual usage, in the caller's transaction.

    Safe to repeat: only a reservation still present is charged. Settling never rolls a
    window: rolling here would start a session window with no request behind it, so usage
    goes to whichever window the bucket has now. A deleted or reset bucket is a no-op.
    """
    if not lease.reservations:
        return
    await _set_lock_timeout(session)
    if usage.usage_missing:
        logger.warning("Billable request reported no usage; charging its quota estimate")
    for reservation in lease.reservations:
        deleted = await session.execute(
            delete(_reservations).where(_reservations.c.id == reservation.reservation_id).returning(_reservations.c.id)
        )
        if deleted.first() is None:
            continue
        amount = actual_amount(reservation.measure, usage, reservation.amount)
        if amount:
            # Core UPDATE so the increment is computed by the database under the row lock,
            # never from a value read earlier.
            await session.execute(
                update(_buckets)
                .where(_buckets.c.id == reservation.bucket_id)
                .values(used=_buckets.c.used + amount, updated_at=func.now())
            )


# --- Estimates ---


def _strip_base64(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("type") == "base64":
            return {k: v for k, v in value.items() if k != "data"}
        return {k: _strip_base64(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_base64(v) for v in value]
    return value


def build_estimate(body: MessagesRequest, llm_model: Any) -> RequestEstimate:
    """Estimate from the whole serialized request (tool definitions and tool results count
    too), leaving out base64 payloads, whose length says little about their token cost."""
    serialized = json.dumps(_strip_base64(body.model_dump(exclude_none=True)), default=str)
    return RequestEstimate(
        input_chars=len(serialized),
        max_tokens=body.max_tokens,
        input_cost_per_1k=Decimal(str(llm_model.input_cost_per_1k or 0)),
        output_cost_per_1k=Decimal(str(llm_model.output_cost_per_1k or 0)),
    )


# --- CRUD ---


async def create_quota(db: AsyncSession, **fields: Any) -> Quota:
    quota = Quota(**fields)
    db.add(quota)
    await db.commit()
    await db.refresh(quota)
    return quota


async def get_quota(db: AsyncSession, quota_id: uuid.UUID) -> Quota | None:
    return await db.get(Quota, quota_id)


async def get_quota_by_name(db: AsyncSession, name: str) -> Quota | None:
    result = await db.execute(select(Quota).where(Quota.name == name))
    return result.scalar_one_or_none()


async def list_quotas(db: AsyncSession, offset: int = 0, limit: int = 50) -> tuple[list[Quota], int]:
    total = (await db.execute(select(func.count()).select_from(Quota))).scalar_one()
    result = await db.execute(select(Quota).order_by(Quota.name).offset(offset).limit(limit))
    return list(result.scalars().all()), total


async def update_quota(db: AsyncSession, quota: Quota, fields: dict[str, Any]) -> Quota:
    """Apply validated fields. A new limit applies immediately because it's read live; a
    change to what buckets mean discards them, in the same transaction."""
    redefined = any(k in _BUCKET_DEFINING_FIELDS and getattr(quota, k) != v for k, v in fields.items())
    for key, value in fields.items():
        setattr(quota, key, value)
    if redefined:
        await db.execute(delete(_buckets).where(_buckets.c.quota_id == quota.id))
    await db.commit()
    await db.refresh(quota)
    return quota


async def rules_using_quota(db: AsyncSession, quota_id: uuid.UUID) -> list[str]:
    result = await db.execute(
        select(Rule.name).where(Rule.action.contains({"type": "apply_quota", "quota_ids": [str(quota_id)]}))
    )
    return list(result.scalars().all())


async def delete_quota(db: AsyncSession, quota: Quota) -> None:
    rules = await rules_using_quota(db, quota.id)
    if rules:
        raise QuotaInUseError(", ".join(sorted(rules)))
    await db.delete(quota)
    await db.commit()


async def reset_quota(db: AsyncSession, quota_id: uuid.UUID) -> int:
    """Discard every bucket of a quota. In-flight requests go uncharged. Returns the count."""
    result = await db.execute(delete(_buckets).where(_buckets.c.quota_id == quota_id).returning(_buckets.c.id))
    count = len(result.all())
    await db.commit()
    return count


# --- Read models ---


def _bucket_view(bucket: Any, quota: Any, in_flight: Decimal, now: datetime) -> dict[str, Any]:
    """A bucket as a reader should see it: an expired window reads as empty, even though the
    row only actually rolls over on its next request."""
    expired = _is_expired(bucket.window_end if bucket is not None else None, now)
    used = Decimal(0) if expired else bucket.used
    in_flight = Decimal(0) if expired else in_flight
    return {
        "quota_id": quota.id,
        "quota_name": quota.name,
        "measure": quota.measure,
        "limit": quota.limit_value,
        "used": used,
        "in_flight": in_flight,
        "remaining": max(quota.limit_value - used - in_flight, Decimal(0)),
        "window_start": None if expired else bucket.window_start,
        "window_end": None if expired else bucket.window_end,
        "dimension_values": bucket.dimension_values if bucket is not None else {},
    }


async def _database_now(db: AsyncSession) -> datetime:
    return (await db.execute(select(func.clock_timestamp()))).scalar_one()


async def list_buckets(
    db: AsyncSession, quota: Quota, offset: int = 0, limit: int = 50
) -> tuple[list[dict[str, Any]], int]:
    total = (
        await db.execute(select(func.count()).select_from(_buckets).where(_buckets.c.quota_id == quota.id))
    ).scalar_one()
    rows = (
        await db.execute(
            select(_buckets)
            .where(_buckets.c.quota_id == quota.id)
            .order_by(_buckets.c.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()
    now = await _database_now(db)
    in_flight = await _in_flight_by_bucket(db, {row.id: row.window_start for row in rows}, now)
    return [_bucket_view(row, quota, in_flight.get(row.id, Decimal(0)), now) for row in rows], total


async def statuses_for_user(db: AsyncSession, user: Any, token_id: uuid.UUID) -> list[dict[str, Any]]:
    """What ``/me/quotas`` shows: a read-only dry run of quota selection for each of the
    user's models. Request content isn't known here, so rules matching on it are best-effort.
    """
    rules = await rules_service.get_active_rules(db)
    if not rules:
        return []
    groups = await rules_service.load_subject_groups(db, rules, user.id)
    models = await model_service.list_user_models(db, user.id)

    targets: dict[tuple[uuid.UUID, str], dict[str, str]] = {}
    quota_ids: set[uuid.UUID] = set()
    per_model: list[tuple[QuotaSubject, set[uuid.UUID]]] = []
    for model in models:
        ctx = rules_service.build_status_context(
            model.name, str(user.id), user.email, str(token_id), groups
        )
        ids = {uuid.UUID(q) for q in collect_quota_ids(rules, ctx)}
        subject = QuotaSubject(str(user.id), user.email, str(model.id), model.name, str(token_id))
        per_model.append((subject, ids))
        quota_ids |= ids
    if not quota_ids:
        return []

    quotas = {
        row.id: row
        for row in (
            await db.execute(select(_quotas).where(_quotas.c.id.in_(list(quota_ids)), _quotas.c.enabled.is_(True)))
        ).all()
    }
    for subject, ids in per_model:
        for quota_id in ids:
            if quota_id in quotas:
                key, values = bucket_identity(tuple(quotas[quota_id].dimensions), subject)
                targets[(quota_id, key)] = values
    if not targets:
        return []

    rows = (
        await db.execute(
            select(_buckets).where(
                _buckets.c.quota_id.in_(list(quotas)),
                _buckets.c.bucket_key.in_([key for _, key in targets]),
            )
        )
    ).all()
    found = {(row.quota_id, row.bucket_key): row for row in rows if (row.quota_id, row.bucket_key) in targets}
    now = await _database_now(db)
    in_flight = await _in_flight_by_bucket(db, {row.id: row.window_start for row in found.values()}, now)

    views = []
    for (quota_id, key), values in targets.items():
        row = found.get((quota_id, key))
        view = _bucket_view(row, quotas[quota_id], in_flight.get(row.id, Decimal(0)) if row else Decimal(0), now)
        view["dimension_values"] = values
        views.append(view)
    return sorted(views, key=lambda v: (v["quota_name"], v["dimension_values"].get("model_name", "")))
