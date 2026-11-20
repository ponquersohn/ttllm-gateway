"""Quota override codes: format, hashing, duration parsing and state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from ttllm.core.quota_override import (
    CODE_PREFIX,
    OverrideState,
    format_duration,
    generate_code,
    hash_code,
    override_state,
    parse_duration,
)

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


def _row(**fields):
    base = dict(revoked_at=None, redeemed_at=None, override_expires_at=None, code_expires_at=NOW + timedelta(hours=1))
    base.update(fields)
    return SimpleNamespace(**base)


def test_codes_are_prefixed_and_unique():
    codes = {generate_code() for _ in range(100)}
    assert len(codes) == 100
    assert all(c.startswith(CODE_PREFIX) and len(c) > 30 for c in codes)


def test_hash_ignores_surrounding_whitespace():
    code = generate_code()
    assert hash_code(code) == hash_code(f"  {code}\n")
    assert len(hash_code(code)) == 64
    assert hash_code(code) != hash_code(generate_code())


@pytest.mark.parametrize(
    "value,seconds",
    [("90", 90), ("45s", 45), ("30m", 1800), ("4h", 14400), ("2d", 172800), ("1w", 604800), (" 3 H ", 10800)],
)
def test_parse_duration(value, seconds):
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["", "h", "-1h", "1.5h", "4x", "4h30m"])
def test_parse_duration_rejects_garbage(value):
    with pytest.raises(ValueError):
        parse_duration(value)


@pytest.mark.parametrize("seconds,text", [(0, "0s"), (60, "1m"), (14400, "4h"), (93600, "1d 2h"), (3661, "1h 1m 1s")])
def test_format_duration(seconds, text):
    assert format_duration(seconds) == text


def test_state_pending_and_code_expired():
    assert override_state(_row(), NOW) is OverrideState.PENDING
    assert override_state(_row(code_expires_at=NOW), NOW) is OverrideState.CODE_EXPIRED


def test_state_active_then_expired_regardless_of_code_expiry():
    redeemed = dict(redeemed_at=NOW - timedelta(hours=2), code_expires_at=NOW - timedelta(hours=1))
    assert override_state(_row(**redeemed, override_expires_at=NOW + timedelta(minutes=1)), NOW) is OverrideState.ACTIVE
    assert override_state(_row(**redeemed, override_expires_at=NOW), NOW) is OverrideState.EXPIRED


def test_revoked_wins():
    row = _row(revoked_at=NOW, redeemed_at=NOW, override_expires_at=NOW + timedelta(hours=1))
    assert override_state(row, NOW) is OverrideState.REVOKED
