"""Token quota enforcement: check (pre-request) and debit (post-request)."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import case, delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.auth import UserGroup
from ttllm.models.quota import LimitScope, TokenLimit, UsageCounter, WindowKind

WINDOW_DURATIONS: dict[WindowKind, timedelta] = {
    WindowKind.FIVE_H: timedelta(hours=5),
    WindowKind.WEEKLY: timedelta(days=7),
    WindowKind.MONTHLY: timedelta(days=30),
}


def _duration_for(limit: TokenLimit) -> timedelta:
    """Resolved window duration: the row's window_seconds if set, else the default."""
    if limit.window_seconds is not None:
        return timedelta(seconds=limit.window_seconds)
    return WINDOW_DURATIONS[limit.window_kind]


async def _resolve_limits(
    db: AsyncSession, user_id: uuid.UUID
) -> dict[WindowKind, tuple[int, timedelta]]:
    """Return {window_kind: (token_cap, duration)} using user > min(group) > global
    precedence. Both cap and duration come from the same winning row."""
    group_ids_q = select(UserGroup.group_id).where(UserGroup.user_id == user_id)
    result = await db.execute(
        select(TokenLimit).where(
            (TokenLimit.scope == LimitScope.USER) & (TokenLimit.user_id == user_id)
            | (TokenLimit.scope == LimitScope.GROUP) & TokenLimit.group_id.in_(group_ids_q)
            | (TokenLimit.scope == LimitScope.GLOBAL)
        )
    )
    all_limits = result.scalars().all()

    by_window: dict[WindowKind, list[TokenLimit]] = {}
    for lim in all_limits:
        by_window.setdefault(lim.window_kind, []).append(lim)

    resolved: dict[WindowKind, tuple[int, timedelta]] = {}
    for wk, lims in by_window.items():
        user_lim = next((l for l in lims if l.scope == LimitScope.USER), None)
        if user_lim:
            resolved[wk] = (user_lim.token_cap, _duration_for(user_lim))
            continue
        group_lims = [l for l in lims if l.scope == LimitScope.GROUP]
        if group_lims:
            winner = min(group_lims, key=lambda l: l.token_cap)
            resolved[wk] = (winner.token_cap, _duration_for(winner))
            continue
        global_lim = next((l for l in lims if l.scope == LimitScope.GLOBAL), None)
        if global_lim:
            resolved[wk] = (global_lim.token_cap, _duration_for(global_lim))
    return resolved


def _make_429(wk: WindowKind, reset_at: datetime, now: datetime) -> HTTPException:
    iso = reset_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    secs = max(1, int((reset_at - now).total_seconds()))
    return HTTPException(
        status_code=429,
        detail={"type": "error", "error": {
            "type": "rate_limit_error",
            "message": f"Token quota exceeded for {wk.value} window. Resets at {iso}.",
        }},
        headers={"retry-after": str(secs), "anthropic-ratelimit-tokens-reset": iso},
    )


async def check_quota(db: AsyncSession, user_id: uuid.UUID) -> None:
    """Raise 429 if any window quota is exhausted. Call pre-request."""
    now = datetime.now(UTC)
    limits = await _resolve_limits(db, user_id)
    if not limits:
        return

    result = await db.execute(
        select(UsageCounter).where(
            UsageCounter.user_id == user_id,
            UsageCounter.window_kind.in_(limits.keys()),
        )
    )
    counters = {c.window_kind: c for c in result.scalars().all()}

    for wk, (cap, duration) in limits.items():
        counter = counters.get(wk)
        if not counter:
            continue  # no row yet — unlimited until first debit seeds it

        if counter.cooldown_until and now < counter.cooldown_until:
            raise _make_429(wk, counter.cooldown_until, now)

        # Lazy reset — window fully elapsed; idempotent guard on window_start < cutoff
        if counter.window_start < now - duration:
            await db.execute(
                update(UsageCounter)
                .where(
                    UsageCounter.user_id == user_id,
                    UsageCounter.window_kind == wk,
                    UsageCounter.window_start < now - duration,
                )
                .values(tokens_used=0, window_start=now, cooldown_until=None)
            )
            continue  # reset clears any over-limit state; admit the request

        if counter.tokens_used >= cap:
            raise _make_429(wk, counter.window_start + duration, now)


async def debit_quota(db: AsyncSession, user_id: uuid.UUID, tokens: int) -> None:
    """Debit tokens post-request. Atomic upsert, committed on completion.

    Called AFTER audit_service.log_request, which has already committed the audit
    row and closed that transaction. The debit therefore opens its own transaction
    and must commit it — otherwise the counter writes are discarded and quota is
    never enforced.
    """
    if tokens <= 0:
        return
    now = datetime.now(UTC)
    limits = await _resolve_limits(db, user_id)
    if not limits:
        return

    for wk, (cap, duration) in limits.items():
        cutoff = now - duration

        # Atomic seed-or-increment with inline lazy reset if window expired.
        stmt = (
            pg_insert(UsageCounter)
            .values(
                user_id=user_id,
                window_kind=wk,
                window_start=now,
                tokens_used=tokens,
                cooldown_until=None,
            )
            .on_conflict_do_update(
                index_elements=["user_id", "window_kind"],
                set_={
                    "tokens_used": case(
                        (UsageCounter.window_start < cutoff, tokens),
                        else_=UsageCounter.tokens_used + tokens,
                    ),
                    "window_start": case(
                        (UsageCounter.window_start < cutoff, now),
                        else_=UsageCounter.window_start,
                    ),
                    "cooldown_until": case(
                        (UsageCounter.window_start < cutoff, None),
                        else_=UsageCounter.cooldown_until,
                    ),
                },
            )
            .returning(UsageCounter.tokens_used, UsageCounter.window_start)
        )
        row = (await db.execute(stmt)).first()
        if row and row[0] > cap:
            next_reset = row[1] + duration
            await db.execute(
                update(UsageCounter)
                .where(UsageCounter.user_id == user_id, UsageCounter.window_kind == wk)
                .values(cooldown_until=next_reset)
            )
    # Commit the counter writes — log_request already committed and closed its
    # transaction before this runs, so the debit owns its own commit.
    await db.commit()


# --- Admin CRUD (used by /admin/usage-limits endpoints) ---


async def create_limit(
    db: AsyncSession,
    scope: LimitScope,
    window_kind: WindowKind,
    token_cap: int,
    user_id: uuid.UUID | None = None,
    group_id: uuid.UUID | None = None,
    window_seconds: int | None = None,
) -> TokenLimit:
    """Create a new token limit row."""
    limit = TokenLimit(
        scope=scope,
        window_kind=window_kind,
        token_cap=token_cap,
        user_id=user_id,
        group_id=group_id,
        window_seconds=window_seconds,
    )
    db.add(limit)
    await db.commit()
    await db.refresh(limit)
    return limit


async def get_limit(db: AsyncSession, limit_id: uuid.UUID) -> TokenLimit | None:
    """Fetch a single token limit by id."""
    return await db.get(TokenLimit, limit_id)


async def list_limits(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
    group_id: uuid.UUID | None = None,
    scope: LimitScope | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[TokenLimit], int]:
    """List token limits with optional filters, returning (rows, total)."""
    query = select(TokenLimit)
    count_query = select(TokenLimit.id)
    if user_id is not None:
        query = query.where(TokenLimit.user_id == user_id)
        count_query = count_query.where(TokenLimit.user_id == user_id)
    if group_id is not None:
        query = query.where(TokenLimit.group_id == group_id)
        count_query = count_query.where(TokenLimit.group_id == group_id)
    if scope is not None:
        query = query.where(TokenLimit.scope == scope)
        count_query = count_query.where(TokenLimit.scope == scope)
    count_result = await db.execute(count_query)
    total = len(count_result.all())
    query = query.offset(offset).limit(limit).order_by(TokenLimit.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_limit(
    db: AsyncSession,
    limit_id: uuid.UUID,
    **fields,
) -> TokenLimit | None:
    """Update a limit's token_cap and/or window_seconds. Clears cooldowns on the
    affected scope so a changed cap/window takes effect immediately."""
    limit = await db.get(TokenLimit, limit_id)
    if not limit:
        return None
    for key, value in fields.items():
        setattr(limit, key, value)
    cooldown_clear = (
        update(UsageCounter)
        .where(UsageCounter.window_kind == limit.window_kind)
        .values(cooldown_until=None)
    )
    if limit.user_id is not None:
        cooldown_clear = cooldown_clear.where(UsageCounter.user_id == limit.user_id)
    await db.execute(cooldown_clear)
    await db.commit()
    await db.refresh(limit)
    return limit


async def delete_limit(db: AsyncSession, limit_id: uuid.UUID) -> bool:
    """Delete a limit. Also clears usage_counter rows for a user-scoped limit so
    a now-deleted limit stops enforcing. Returns True if a row was deleted."""
    limit = await db.get(TokenLimit, limit_id)
    if not limit:
        return False
    # For user-scoped limits, clear the matching counter so enforcement stops cleanly.
    if limit.user_id is not None:
        await db.execute(
            delete(UsageCounter).where(
                UsageCounter.user_id == limit.user_id,
                UsageCounter.window_kind == limit.window_kind,
            )
        )
    await db.delete(limit)
    await db.commit()
    return True
