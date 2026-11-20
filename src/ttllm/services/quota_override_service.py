"""Quota override codes: issue, preview, redeem, revoke, and the per-request lookup.

Issuing, redeeming and revoking don't commit: the caller writes the admin audit entry
in the same transaction (``admin_audit_service.log`` commits), so the change and its
audit record land together.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.core.quota_override import OverrideState, generate_code, hash_code, override_state
from ttllm.models.quota import QuotaOverride
from ttllm.models.user import User

INVALID_CODE_MESSAGE = "Invalid, expired or already used override code"


class InvalidOverrideCodeError(Exception):
    pass


async def _now(db: AsyncSession) -> datetime:
    return (await db.execute(select(func.clock_timestamp()))).scalar_one()


async def issue(
    db: AsyncSession, issuer: User, override_seconds: int, code_ttl_seconds: int, reason: str
) -> tuple[QuotaOverride, str]:
    """Create a pending override. Returns the row and the plaintext code (shown once)."""
    code = generate_code()
    now = await _now(db)
    row = QuotaOverride(
        code_hash=hash_code(code),
        created_by=issuer.id,
        created_by_email=issuer.email,
        reason=reason,
        override_duration_seconds=override_seconds,
        code_expires_at=now + timedelta(seconds=code_ttl_seconds),
    )
    db.add(row)
    await db.flush()
    await db.refresh(row)
    return row, code


def _redeemable(code: str, now: Any):
    return and_(
        QuotaOverride.code_hash == hash_code(code),
        QuotaOverride.redeemed_at.is_(None),
        QuotaOverride.revoked_at.is_(None),
        QuotaOverride.code_expires_at > now,
    )


async def lookup(db: AsyncSession, code: str) -> QuotaOverride | None:
    """The pending override behind ``code``, or None if it can't be redeemed."""
    result = await db.execute(select(QuotaOverride).where(_redeemable(code, func.clock_timestamp())))
    return result.scalar_one_or_none()


async def redeem(db: AsyncSession, code: str, user: User) -> QuotaOverride:
    """Redeem ``code`` for ``user``. One conditional UPDATE, so a code can't be used twice
    even by concurrent requests. The override clock starts now."""
    now = func.clock_timestamp()
    result = await db.execute(
        update(QuotaOverride)
        .where(_redeemable(code, now))
        .values(
            redeemed_by=user.id,
            redeemed_at=now,
            override_expires_at=now
            + func.make_interval(0, 0, 0, 0, 0, 0, QuotaOverride.override_duration_seconds),
        )
        .returning(QuotaOverride.id)
    )
    override_id = result.scalar_one_or_none()
    if override_id is None:
        raise InvalidOverrideCodeError(INVALID_CODE_MESSAGE)
    row = await db.get(QuotaOverride, override_id, populate_existing=True)
    assert row is not None
    return row


async def active_override_for_user(db: AsyncSession, user_id: uuid.UUID) -> QuotaOverride | None:
    """The user's longest-running active override, if any. Checked on every quota-bound request."""
    result = await db.execute(
        select(QuotaOverride)
        .where(
            QuotaOverride.redeemed_by == user_id,
            QuotaOverride.revoked_at.is_(None),
            QuotaOverride.override_expires_at > func.clock_timestamp(),
        )
        .order_by(QuotaOverride.override_expires_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def list_active_for_user(db: AsyncSession, user_id: uuid.UUID) -> list[QuotaOverride]:
    result = await db.execute(
        select(QuotaOverride)
        .where(
            QuotaOverride.redeemed_by == user_id,
            QuotaOverride.revoked_at.is_(None),
            QuotaOverride.override_expires_at > func.clock_timestamp(),
        )
        .order_by(QuotaOverride.override_expires_at.desc())
    )
    return list(result.scalars().all())


def _state_filter(state: OverrideState, now: Any):
    not_revoked = QuotaOverride.revoked_at.is_(None)
    match state:
        case OverrideState.REVOKED:
            return QuotaOverride.revoked_at.is_not(None)
        case OverrideState.PENDING:
            return and_(not_revoked, QuotaOverride.redeemed_at.is_(None), QuotaOverride.code_expires_at > now)
        case OverrideState.CODE_EXPIRED:
            return and_(not_revoked, QuotaOverride.redeemed_at.is_(None), QuotaOverride.code_expires_at <= now)
        case OverrideState.ACTIVE:
            return and_(not_revoked, QuotaOverride.override_expires_at > now)
        case OverrideState.EXPIRED:
            return and_(not_revoked, QuotaOverride.override_expires_at <= now)


async def list_overrides(
    db: AsyncSession, state: OverrideState | None = None, offset: int = 0, limit: int = 50
) -> tuple[list[dict[str, Any]], int]:
    """Override rows as admin views (with state and redeemer email), newest first."""
    now = await _now(db)
    where = _state_filter(state, now) if state else None
    count_q = select(func.count()).select_from(QuotaOverride)
    rows_q = (
        select(QuotaOverride, User.email)
        .outerjoin(User, User.id == QuotaOverride.redeemed_by)
        .order_by(QuotaOverride.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if where is not None:
        count_q = count_q.where(where)
        rows_q = rows_q.where(where)
    total = (await db.execute(count_q)).scalar_one()
    rows = (await db.execute(rows_q)).all()
    return [view(row, now, redeemed_by_email=email) for row, email in rows], total


async def get_override(db: AsyncSession, override_id: uuid.UUID) -> QuotaOverride | None:
    return await db.get(QuotaOverride, override_id)


async def revoke(db: AsyncSession, override_id: uuid.UUID, actor: User) -> QuotaOverride | None:
    """Revoke a pending code or end an active override. None if not found or already revoked."""
    result = await db.execute(
        update(QuotaOverride)
        .where(QuotaOverride.id == override_id, QuotaOverride.revoked_at.is_(None))
        .values(revoked_at=func.clock_timestamp(), revoked_by=actor.id)
        .returning(QuotaOverride.id)
    )
    if result.scalar_one_or_none() is None:
        return None
    return await db.get(QuotaOverride, override_id, populate_existing=True)


def view(row: QuotaOverride, now: datetime, redeemed_by_email: str | None = None) -> dict[str, Any]:
    return {
        "id": row.id,
        "state": override_state(row, now).value,
        "reason": row.reason,
        "created_by_email": row.created_by_email,
        "override_duration_seconds": row.override_duration_seconds,
        "code_expires_at": row.code_expires_at,
        "redeemed_by_email": redeemed_by_email,
        "redeemed_at": row.redeemed_at,
        "override_expires_at": row.override_expires_at,
        "revoked_at": row.revoked_at,
        "created_at": row.created_at,
    }


async def views_for(db: AsyncSession, rows: list[QuotaOverride], redeemed_by_email: str | None = None):
    now = await _now(db)
    return [view(r, now, redeemed_by_email=redeemed_by_email) for r in rows]

