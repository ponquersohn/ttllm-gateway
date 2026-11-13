"""Service for the OAuth 2.0 device code flow (RFC 8628).

Rows hold only hashes of the device code and user code. Resolution (approve/deny)
happens from an authenticated session in the UI; the CLI polls with the device code.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.core import device_flow
from ttllm.core.device_flow import PollOutcome, PollState
from ttllm.models.device_auth import DeviceAuthorization
from ttllm.models.user import User

_USER_CODE_ATTEMPTS = 5


async def create(
    db: AsyncSession,
    *,
    client_name: str | None,
    requester_ip: str | None,
    requester_user_agent: str | None,
) -> tuple[DeviceAuthorization, str, str]:
    """Create a pending authorization. Returns (row, raw device_code, formatted user_code)."""
    now = datetime.now(UTC)
    # Lazy cleanup: nothing else sweeps this table.
    await db.execute(delete(DeviceAuthorization).where(DeviceAuthorization.expires_at < now))

    device_code = device_flow.generate_device_code()
    for _ in range(_USER_CODE_ATTEMPTS):
        user_code = device_flow.generate_user_code()
        user_code_hash = device_flow.hash_code(device_flow.normalize_user_code(user_code))
        clash = await db.execute(
            select(DeviceAuthorization.id).where(DeviceAuthorization.user_code_hash == user_code_hash)
        )
        if clash.first() is None:
            break
    else:  # pragma: no cover - 2.5e10 code space
        raise RuntimeError("Could not allocate a unique user code")

    row = DeviceAuthorization(
        device_code_hash=device_flow.hash_code(device_code),
        user_code_hash=user_code_hash,
        status=device_flow.STATUS_PENDING,
        client_name=client_name[:255] if client_name else None,
        requester_ip=requester_ip[:64] if requester_ip else None,
        requester_user_agent=requester_user_agent[:512] if requester_user_agent else None,
        interval_seconds=device_flow.DEFAULT_INTERVAL,
        expires_at=now + device_flow.DEVICE_CODE_TTL,
    )
    db.add(row)
    await db.flush()
    return row, device_code, user_code


async def get_pending_by_user_code(db: AsyncSession, user_code: str) -> DeviceAuthorization | None:
    """Return the pending, unexpired authorization for a user code, or None."""
    user_code_hash = device_flow.hash_code(device_flow.normalize_user_code(user_code))
    result = await db.execute(
        select(DeviceAuthorization).where(
            DeviceAuthorization.user_code_hash == user_code_hash,
            DeviceAuthorization.status == device_flow.STATUS_PENDING,
            DeviceAuthorization.expires_at > datetime.now(UTC),
        ).with_for_update()
    )
    return result.scalar_one_or_none()


async def resolve(db: AsyncSession, user_code: str, user: User, approve: bool) -> bool:
    """Approve or deny a pending authorization on behalf of `user`. False if not found."""
    row = await get_pending_by_user_code(db, user_code)
    if row is None:
        return False
    row.status = device_flow.STATUS_APPROVED if approve else device_flow.STATUS_DENIED
    row.user_id = user.id if approve else None
    await db.flush()
    return True


async def poll(db: AsyncSession, device_code: str) -> tuple[PollOutcome | None, User | None]:
    """Evaluate a token poll. Returns (None, None) for an unknown device code.

    Terminal outcomes (approved/denied/expired) delete the row, so a device code
    can mint tokens at most once. The row is locked so concurrent polls can't
    both observe APPROVED. Caller commits.
    """
    result = await db.execute(
        select(DeviceAuthorization)
        .where(DeviceAuthorization.device_code_hash == device_flow.hash_code(device_code))
        .with_for_update()
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None, None

    now = datetime.now(UTC)
    outcome = device_flow.evaluate_poll(
        PollState(
            status=row.status,
            expires_at=row.expires_at,
            interval_seconds=row.interval_seconds,
            last_polled_at=row.last_polled_at,
        ),
        now,
    )

    user: User | None = None
    if outcome == PollOutcome.APPROVED and row.user_id is not None:
        user = await db.get(User, row.user_id)

    if outcome in (PollOutcome.APPROVED, PollOutcome.ACCESS_DENIED, PollOutcome.EXPIRED_TOKEN):
        await db.delete(row)
    else:
        if outcome == PollOutcome.SLOW_DOWN:
            row.interval_seconds += device_flow.SLOW_DOWN_INCREMENT
        row.last_polled_at = now
    await db.flush()

    if outcome == PollOutcome.APPROVED and user is None:
        # Approving user was deleted between approval and poll.
        return PollOutcome.ACCESS_DENIED, None
    return outcome, user


def expires_in(row: DeviceAuthorization) -> int:
    return max(0, int((row.expires_at - datetime.now(UTC)) / timedelta(seconds=1)))
