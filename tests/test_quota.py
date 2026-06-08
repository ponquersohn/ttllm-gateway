"""Unit tests for the quota service, including custom window durations and precedence resolution."""

from datetime import UTC, datetime, timedelta
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from ttllm.models.quota import LimitScope, TokenLimit, UsageCounter, WindowKind
from ttllm.services import quota_service


def test_duration_for_default():
    limit = TokenLimit(
        window_kind=WindowKind.FIVE_H,
        token_cap=1000,
        window_seconds=None,
    )
    duration = quota_service._duration_for(limit)
    assert duration == timedelta(hours=5)


def test_duration_for_custom():
    limit = TokenLimit(
        window_kind=WindowKind.FIVE_H,
        token_cap=1000,
        window_seconds=3600,
    )
    duration = quota_service._duration_for(limit)
    assert duration == timedelta(seconds=3600)


@pytest.mark.asyncio
async def test_resolve_limits_precedence():
    db = AsyncMock()
    user_id = uuid.uuid4()

    # Create dummy limits simulating: user, group (2 limits), global
    user_lim = TokenLimit(
        scope=LimitScope.USER,
        user_id=user_id,
        window_kind=WindowKind.FIVE_H,
        token_cap=5000,
        window_seconds=1800,
    )
    group_lim_1 = TokenLimit(
        scope=LimitScope.GROUP,
        group_id=uuid.uuid4(),
        window_kind=WindowKind.FIVE_H,
        token_cap=10000,
        window_seconds=3600,
    )
    group_lim_2 = TokenLimit(
        scope=LimitScope.GROUP,
        group_id=uuid.uuid4(),
        window_kind=WindowKind.FIVE_H,
        token_cap=8000,
        window_seconds=7200,
    )
    global_lim = TokenLimit(
        scope=LimitScope.GLOBAL,
        window_kind=WindowKind.FIVE_H,
        token_cap=20000,
        window_seconds=None,
    )

    # Mock execute return values
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        user_lim, group_lim_1, group_lim_2, global_lim
    ]
    db.execute.return_value = mock_result

    resolved = await quota_service._resolve_limits(db, user_id)
    # USER scope should win
    assert WindowKind.FIVE_H in resolved
    cap, duration = resolved[WindowKind.FIVE_H]
    assert cap == 5000
    assert duration == timedelta(seconds=1800)


@pytest.mark.asyncio
async def test_resolve_limits_group_precedence():
    db = AsyncMock()
    user_id = uuid.uuid4()

    # Simulating only group and global limits (user limit is missing).
    # The group limit with the minimum token_cap should win.
    group_lim_1 = TokenLimit(
        scope=LimitScope.GROUP,
        group_id=uuid.uuid4(),
        window_kind=WindowKind.FIVE_H,
        token_cap=10000,
        window_seconds=3600,
    )
    group_lim_2 = TokenLimit(
        scope=LimitScope.GROUP,
        group_id=uuid.uuid4(),
        window_kind=WindowKind.FIVE_H,
        token_cap=8000,
        window_seconds=7200,
    )
    global_lim = TokenLimit(
        scope=LimitScope.GLOBAL,
        window_kind=WindowKind.FIVE_H,
        token_cap=20000,
        window_seconds=None,
    )

    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [
        group_lim_1, group_lim_2, global_lim
    ]
    db.execute.return_value = mock_result

    resolved = await quota_service._resolve_limits(db, user_id)
    # The min group limit (cap=8000) should win
    assert WindowKind.FIVE_H in resolved
    cap, duration = resolved[WindowKind.FIVE_H]
    assert cap == 8000
    assert duration == timedelta(seconds=7200)


@pytest.mark.asyncio
async def test_check_quota_within_limit(monkeypatch):
    db = AsyncMock()
    user_id = uuid.uuid4()

    # Mock _resolve_limits (monkeypatch auto-restores after the test)
    mock_limits = {
        WindowKind.FIVE_H: (1000, timedelta(seconds=3600))
    }
    monkeypatch.setattr(quota_service, "_resolve_limits", AsyncMock(return_value=mock_limits))

    # Mock query for UsageCounter
    counter = UsageCounter(
        user_id=user_id,
        window_kind=WindowKind.FIVE_H,
        window_start=datetime.now(UTC) - timedelta(minutes=10),
        tokens_used=500,
        cooldown_until=None,
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [counter]
    db.execute.return_value = mock_result

    # Should not raise any exception
    await quota_service.check_quota(db, user_id)


@pytest.mark.asyncio
async def test_check_quota_exceeded(monkeypatch):
    db = AsyncMock()
    user_id = uuid.uuid4()

    mock_limits = {
        WindowKind.FIVE_H: (1000, timedelta(seconds=3600))
    }
    monkeypatch.setattr(quota_service, "_resolve_limits", AsyncMock(return_value=mock_limits))

    counter = UsageCounter(
        user_id=user_id,
        window_kind=WindowKind.FIVE_H,
        window_start=datetime.now(UTC) - timedelta(minutes=10),
        tokens_used=1200,
        cooldown_until=None,
    )
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = [counter]
    db.execute.return_value = mock_result

    with pytest.raises(HTTPException) as exc_info:
        await quota_service.check_quota(db, user_id)
    assert exc_info.value.status_code == 429
