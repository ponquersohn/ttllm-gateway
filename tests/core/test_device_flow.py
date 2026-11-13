"""Unit tests for the pure device-code-flow logic (core/device_flow.py)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from ttllm.core import device_flow
from ttllm.core.device_flow import PollOutcome, PollState

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=UTC)


def _state(status="pending", expires_in=600, interval=5, last_polled_ago=None) -> PollState:
    return PollState(
        status=status,
        expires_at=NOW + timedelta(seconds=expires_in),
        interval_seconds=interval,
        last_polled_at=None if last_polled_ago is None else NOW - timedelta(seconds=last_polled_ago),
    )


def test_user_code_format_and_alphabet():
    for _ in range(200):
        code = device_flow.generate_user_code()
        assert re.fullmatch(r"[BCDFGHJKLMNPQRSTVWXZ]{4}-[BCDFGHJKLMNPQRSTVWXZ]{4}", code), code


def test_device_code_is_long_and_unique():
    codes = {device_flow.generate_device_code() for _ in range(50)}
    assert len(codes) == 50
    assert all(len(c) >= 40 for c in codes)


@pytest.mark.parametrize("raw", ["WDJB-MJHT", "wdjb-mjht", " wdjb mjht ", "WDJBMJHT"])
def test_normalize_user_code(raw):
    assert device_flow.normalize_user_code(raw) == "WDJBMJHT"


def test_hash_code_is_sha256_hex():
    h = device_flow.hash_code("abc")
    assert len(h) == 64 and h == device_flow.hash_code("abc") != device_flow.hash_code("abd")


def test_first_poll_is_pending():
    assert device_flow.evaluate_poll(_state(), NOW) == PollOutcome.AUTHORIZATION_PENDING


def test_poll_after_interval_is_pending():
    assert device_flow.evaluate_poll(_state(last_polled_ago=5), NOW) == PollOutcome.AUTHORIZATION_PENDING


def test_poll_too_fast_is_slow_down():
    assert device_flow.evaluate_poll(_state(last_polled_ago=2), NOW) == PollOutcome.SLOW_DOWN


def test_approved_ignores_rate():
    assert device_flow.evaluate_poll(_state("approved", last_polled_ago=0), NOW) == PollOutcome.APPROVED


def test_denied():
    assert device_flow.evaluate_poll(_state("denied"), NOW) == PollOutcome.ACCESS_DENIED


@pytest.mark.parametrize("status", ["pending", "approved", "denied"])
def test_expiry_wins(status):
    assert device_flow.evaluate_poll(_state(status, expires_in=0), NOW) == PollOutcome.EXPIRED_TOKEN
