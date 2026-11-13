"""Pure logic for the OAuth 2.0 Device Authorization Grant (RFC 8628).

No framework dependencies: code generation, normalization, hashing and the
poll-outcome decision table live here so they can be tested in isolation.
"""

from __future__ import annotations

import enum
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta

DEVICE_CODE_TTL = timedelta(minutes=10)
DEFAULT_INTERVAL = 5
SLOW_DOWN_INCREMENT = 5

# RFC 8628 §6.1: consonants only, avoids ambiguous characters and accidental words.
USER_CODE_ALPHABET = "BCDFGHJKLMNPQRSTVWXZ"
USER_CODE_LENGTH = 8

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_DENIED = "denied"


class PollOutcome(str, enum.Enum):
    """Result of a token poll. Non-APPROVED values are the RFC 8628 error codes."""

    APPROVED = "approved"
    AUTHORIZATION_PENDING = "authorization_pending"
    SLOW_DOWN = "slow_down"
    ACCESS_DENIED = "access_denied"
    EXPIRED_TOKEN = "expired_token"


@dataclass(frozen=True)
class PollState:
    status: str
    expires_at: datetime
    interval_seconds: int
    last_polled_at: datetime | None


def generate_device_code() -> str:
    return secrets.token_urlsafe(32)


def generate_user_code() -> str:
    raw = "".join(secrets.choice(USER_CODE_ALPHABET) for _ in range(USER_CODE_LENGTH))
    return format_user_code(raw)


def format_user_code(raw: str) -> str:
    half = len(raw) // 2
    return f"{raw[:half]}-{raw[half:]}"


def normalize_user_code(code: str) -> str:
    """Uppercase and strip separators so `wdjb mjht` and `WDJB-MJHT` match."""
    return "".join(ch for ch in code.upper() if ch.isalnum())


def hash_code(code: str) -> str:
    """SHA-256 hex digest for DB storage (device codes and normalized user codes)."""
    return hashlib.sha256(code.encode()).hexdigest()


def evaluate_poll(state: PollState, now: datetime) -> PollOutcome:
    """Decide the response to a token poll.

    Expiry wins over everything; a resolved (approved/denied) authorization is
    reported immediately regardless of poll rate; only a still-pending one is
    subject to `slow_down`.
    """
    if state.expires_at <= now:
        return PollOutcome.EXPIRED_TOKEN
    if state.status == STATUS_APPROVED:
        return PollOutcome.APPROVED
    if state.status == STATUS_DENIED:
        return PollOutcome.ACCESS_DENIED
    if state.last_polled_at is not None and now - state.last_polled_at < timedelta(seconds=state.interval_seconds):
        return PollOutcome.SLOW_DOWN
    return PollOutcome.AUTHORIZATION_PENDING
