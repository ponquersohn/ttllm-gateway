"""Quota override codes. Pure logic, no framework dependencies.

A code is issued with two lifetimes: how long it can be redeemed, and how long the override
it grants lasts once redeemed. The override clock starts at redemption, not at issue.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime
from enum import Enum
from typing import Any

CODE_PREFIX = "ttqo_"

MIN_DURATION_SECONDS = 60
MAX_OVERRIDE_SECONDS = 30 * 24 * 3600
MAX_CODE_TTL_SECONDS = 7 * 24 * 3600

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw]?)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class OverrideState(str, Enum):
    PENDING = "pending"  # issued, redeemable
    ACTIVE = "active"  # redeemed, in effect
    EXPIRED = "expired"  # redeemed, override over
    CODE_EXPIRED = "code_expired"  # never redeemed in time
    REVOKED = "revoked"


def generate_code() -> str:
    return CODE_PREFIX + secrets.token_urlsafe(24)


def normalize_code(code: str) -> str:
    return code.strip()


def hash_code(code: str) -> str:
    """SHA-256 hex digest of the normalized code, for DB storage."""
    return hashlib.sha256(normalize_code(code).encode()).hexdigest()


def override_state(row: Any, now: datetime) -> OverrideState:
    """Where an override row stands at ``now``. Revocation wins over everything else."""
    if row.revoked_at is not None:
        return OverrideState.REVOKED
    if row.redeemed_at is None:
        return OverrideState.PENDING if row.code_expires_at > now else OverrideState.CODE_EXPIRED
    if row.override_expires_at is not None and row.override_expires_at > now:
        return OverrideState.ACTIVE
    return OverrideState.EXPIRED


def parse_duration(value: str) -> int:
    """Parse ``90``, ``30m``, ``4h``, ``2d``, ``1w`` into seconds. Raises ValueError."""
    match = _DURATION_RE.match(value)
    if not match:
        raise ValueError(f"Invalid duration '{value}': use a number with an optional s/m/h/d/w suffix")
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2).lower()]


def format_duration(seconds: int) -> str:
    """Compact human form: ``4h``, ``1d 2h``, ``45m``."""
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            count, seconds = divmod(seconds, size)
            parts.append(f"{count}{unit}")
    return " ".join(parts) or "0s"
