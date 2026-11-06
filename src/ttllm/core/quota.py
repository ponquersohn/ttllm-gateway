"""Quota core logic: windows, bucket identity, estimates, admission. No framework dependencies.

A *quota* is an admin-defined limit on one measure over an interval. A *bucket* is one
counter of that quota, identified by the resolved values of the quota's dimensions (one
bucket per user, per user+model, ...). All I/O, locking and clock reads live in
``services/quota_service.py``; everything here is a pure function of its arguments.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from ttllm.core.rules import render_template

# Approximate chars-per-token, matching the rules engine's ``count_tokens`` heuristic.
_CHARS_PER_TOKEN = 4

# Completed requests, and client disconnects whose stream was drained to final usage.
BILLABLE_STATUS_CODES = frozenset({200, 499})

DEFAULT_EXCEEDED_STATUS = 429
# A request that can never fit (strict estimate over the whole limit): retrying can't help,
# and SDKs retry 429s automatically, so it's reported as a bad request instead.
NEVER_FITS_STATUS = 400


class Measure(str, Enum):
    REQUESTS = "requests"
    TOKENS = "tokens"
    COST = "cost"


class WindowType(str, Enum):
    CALENDAR = "calendar"
    SESSION = "session"


class WindowUnit(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class Enforcement(str, Enum):
    SOFT = "soft"
    STRICT = "strict"


class Dimension(str, Enum):
    USER = "user"
    MODEL = "model"
    TOKEN = "token"


# --- Windows ---


@dataclass(frozen=True)
class WindowSpec:
    type: WindowType
    unit: WindowUnit | None = None
    duration_seconds: int | None = None
    timezone: str = "UTC"


def compute_window(spec: WindowSpec, now: datetime) -> tuple[datetime, datetime]:
    """Return the ``(start, end)`` of the window a bucket should use at ``now`` (both UTC).

    Calendar windows are aligned to the wall clock in ``spec.timezone``, so a daily window
    is 23 or 25 hours long across a DST change. Session windows start at ``now``.
    """
    if spec.type == WindowType.SESSION:
        if not spec.duration_seconds:
            raise ValueError("session window requires duration_seconds")
        return now.astimezone(timezone.utc), (now + timedelta(seconds=spec.duration_seconds)).astimezone(timezone.utc)

    tz = ZoneInfo(spec.timezone)
    local = now.astimezone(tz)

    if spec.unit == WindowUnit.HOUR:
        # Absolute arithmetic: wall-clock "next hour" is ambiguous across a DST fold.
        start = local.replace(minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        return start, start + timedelta(hours=1)

    midnight = datetime(local.year, local.month, local.day, tzinfo=tz)
    if spec.unit == WindowUnit.DAY:
        start_local, end_local = midnight, midnight + timedelta(days=1)
    elif spec.unit == WindowUnit.WEEK:
        start_local = midnight - timedelta(days=local.weekday())  # Monday
        end_local = start_local + timedelta(days=7)
    elif spec.unit == WindowUnit.MONTH:
        start_local = midnight.replace(day=1)
        end_local = (start_local + timedelta(days=32)).replace(day=1)
    else:
        raise ValueError(f"calendar window requires a unit, got {spec.unit!r}")
    # Aware-datetime arithmetic above is wall-clock within one tz, which is what we want;
    # astimezone then applies whichever UTC offset is in force on that local date.
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def seconds_until(end: datetime, now: datetime) -> int:
    """Whole seconds until ``end``, rounded up so a client never retries too early."""
    return max(0, math.ceil((end - now).total_seconds()))


# --- Bucket identity ---


@dataclass(frozen=True)
class QuotaSubject:
    """Server-derived identity of a request. Nothing here is client-controlled.

    ``model_id`` is the *resolved* model: ``model_name`` from the request can be any string
    matching a model's ``match_pattern``, so keying on it would let clients spawn buckets.
    """

    user_id: str
    user_email: str
    model_id: str
    model_name: str
    token_id: str


def bucket_identity(dimensions: tuple[str, ...], subject: QuotaSubject) -> tuple[str, dict[str, str]]:
    """Return ``(bucket_key, display_values)`` for a quota's dimensions and this subject.

    The key hashes canonical JSON of IDs, so names can contain any character and a rename
    keeps the same bucket. ``display_values`` adds readable names for admin output.
    """
    ids: dict[str, str] = {}
    display: dict[str, str] = {}
    for dimension in dimensions:
        if dimension == Dimension.USER:
            ids["user"] = subject.user_id
            display.update(user_id=subject.user_id, user_email=subject.user_email)
        elif dimension == Dimension.MODEL:
            ids["model"] = subject.model_id
            display.update(model_id=subject.model_id, model_name=subject.model_name)
        elif dimension == Dimension.TOKEN:
            ids["token"] = subject.token_id
            display.update(token_id=subject.token_id)
        else:
            raise ValueError(f"unknown quota dimension: {dimension}")
    canonical = json.dumps(ids, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), display


# --- Estimates and actuals ---


@dataclass(frozen=True)
class RequestEstimate:
    input_chars: int
    max_tokens: int
    input_cost_per_1k: Decimal = Decimal("0")
    output_cost_per_1k: Decimal = Decimal("0")


def estimate_amount(measure: Measure | str, estimate: RequestEstimate) -> Decimal:
    """Upper-ish estimate of what a request will consume, reserved until it settles."""
    if measure == Measure.REQUESTS:
        return Decimal(1)
    input_tokens = estimate.input_chars // _CHARS_PER_TOKEN
    if measure == Measure.TOKENS:
        return Decimal(input_tokens + estimate.max_tokens)
    if measure == Measure.COST:
        return (
            Decimal(input_tokens) * estimate.input_cost_per_1k + Decimal(estimate.max_tokens) * estimate.output_cost_per_1k
        ) / 1000
    raise ValueError(f"unknown quota measure: {measure}")


@dataclass(frozen=True)
class ActualUsage:
    status_code: int
    tokens: int = 0
    cost: Decimal = Decimal("0")

    @property
    def billable(self) -> bool:
        return self.status_code in BILLABLE_STATUS_CODES

    @property
    def usage_missing(self) -> bool:
        """A billable exchange that reported no tokens: the provider didn't surface usage."""
        return self.billable and self.tokens == 0


def actual_amount(measure: Measure | str, usage: ActualUsage, reserved: Decimal) -> Decimal:
    """What to charge at settlement.

    Failed requests don't consume ``requests``, but any tokens or cost they did report were
    paid for, so those are still charged. When a billable exchange reported no usage at all,
    charge the reservation rather than letting it pass for free.
    """
    if measure == Measure.REQUESTS:
        return Decimal(1) if usage.billable else Decimal(0)
    if usage.usage_missing:
        return reserved
    if measure == Measure.TOKENS:
        return Decimal(usage.tokens)
    if measure == Measure.COST:
        return usage.cost
    raise ValueError(f"unknown quota measure: {measure}")


# --- Admission ---


def admits(
    enforcement: Enforcement | str,
    used: Decimal,
    in_flight: Decimal,
    estimate: Decimal,
    limit: Decimal,
) -> bool:
    """Whether a new request fits.

    ``soft`` admits while anything is left, so large ``max_tokens`` values aren't rejected
    near the limit; overshoot is bounded to about one request because every admitted request
    immediately counts as in-flight for the next. ``strict`` never exceeds the limit, at the
    price of rejecting requests whose worst-case estimate doesn't fit.
    """
    if enforcement == Enforcement.STRICT:
        return used + in_flight + estimate <= limit
    return used + in_flight < limit


# --- Denials ---


@dataclass(frozen=True)
class QuotaDenial:
    quota_id: str
    name: str
    measure: str
    limit: Decimal
    used: Decimal
    in_flight: Decimal
    # None when waiting can't help: a strict estimate larger than the whole limit.
    retry_after: int | None
    on_exceeded: dict[str, Any]


@dataclass(frozen=True)
class DenialDecision:
    primary: QuotaDenial
    # Set only when every exceeded quota asks for a reroute.
    reroute_target: str | None
    retry_after: int | None


def decide_denial(denials: list[QuotaDenial]) -> DenialDecision:
    """Pick the response for a denied request. Any blocking quota beats any reroute."""
    if not denials:
        raise ValueError("decide_denial requires at least one denial")
    by_name = sorted(denials, key=lambda d: d.name)
    blocking = [d for d in by_name if d.on_exceeded.get("action", "block") != "reroute"]
    waits = [d.retry_after for d in denials if d.retry_after is not None]
    retry_after = max(waits) if waits else None
    if blocking:
        primary = max(blocking, key=lambda d: d.retry_after if d.retry_after is not None else -1)
        return DenialDecision(primary=primary, reroute_target=None, retry_after=retry_after)
    return DenialDecision(primary=by_name[0], reroute_target=by_name[0].on_exceeded["target"], retry_after=retry_after)


def format_amount(value: Decimal) -> str:
    """Render a NUMERIC(38,18) without trailing zeros or scientific notation."""
    return format(value.normalize(), "f")


def never_fits(denial: QuotaDenial) -> bool:
    """Waiting can't help: the request's own estimate is larger than the whole limit."""
    return denial.retry_after is None


def denial_status(denial: QuotaDenial) -> int:
    """An admin-set ``status_code`` always wins; otherwise 429, or 400 if it can never fit."""
    explicit = denial.on_exceeded.get("status_code")
    if explicit:
        return explicit
    return NEVER_FITS_STATUS if never_fits(denial) else DEFAULT_EXCEEDED_STATUS


def denial_error_type(denial: QuotaDenial) -> str:
    return "invalid_request_error" if denial_status(denial) == NEVER_FITS_STATUS else "rate_limit_error"


def denial_message(denial: QuotaDenial) -> str:
    if denial.on_exceeded.get("message"):
        template = denial.on_exceeded["message"]
    elif never_fits(denial):
        template = (
            "Request can never fit quota '{{quota.name}}': its worst-case estimate is larger than the "
            "whole {{quota.limit}} {{quota.measure}} limit. Lower max_tokens or shorten the request."
        )
    else:
        template = "Quota '{{quota.name}}' exceeded ({{quota.used}}/{{quota.limit}} {{quota.measure}})."
    namespace = {
        "quota": {
            "name": denial.name,
            "measure": denial.measure,
            "used": format_amount(denial.used),
            "limit": format_amount(denial.limit),
            "in_flight": format_amount(denial.in_flight),
            "retry_after": "" if denial.retry_after is None else denial.retry_after,
        }
    }
    return render_template(template, namespace)
