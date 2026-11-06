"""Pure quota logic: windows, bucket identity, estimates, admission, denial decisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ttllm.core.quota import (
    ActualUsage,
    Enforcement,
    Measure,
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
    decide_denial,
    denial_error_type,
    denial_message,
    denial_status,
    estimate_amount,
    format_amount,
    seconds_until,
)
from ttllm.schemas.admin import QuotaDefinition
from ttllm.services.quota_service import _strip_base64

UTC = timezone.utc


def _calendar(unit: WindowUnit, tz: str = "UTC") -> WindowSpec:
    return WindowSpec(type=WindowType.CALENDAR, unit=unit, timezone=tz)


class TestCalendarWindows:
    def test_hour(self):
        now = datetime(2026, 3, 10, 14, 37, 12, 5, tzinfo=UTC)
        assert compute_window(_calendar(WindowUnit.HOUR), now) == (
            datetime(2026, 3, 10, 14, tzinfo=UTC),
            datetime(2026, 3, 10, 15, tzinfo=UTC),
        )

    def test_day_in_timezone(self):
        # 03:00 UTC on the 10th is still the 9th in New York (UTC-5 in winter).
        now = datetime(2026, 2, 10, 3, tzinfo=UTC)
        start, end = compute_window(_calendar(WindowUnit.DAY, "America/New_York"), now)
        assert start == datetime(2026, 2, 9, 5, tzinfo=UTC)
        assert end == datetime(2026, 2, 10, 5, tzinfo=UTC)

    def test_day_is_23_hours_on_spring_forward(self):
        now = datetime(2026, 3, 8, 12, tzinfo=UTC)
        start, end = compute_window(_calendar(WindowUnit.DAY, "America/New_York"), now)
        assert end - start == timedelta(hours=23)

    def test_day_is_25_hours_on_fall_back(self):
        now = datetime(2026, 11, 1, 12, tzinfo=UTC)
        start, end = compute_window(_calendar(WindowUnit.DAY, "America/New_York"), now)
        assert end - start == timedelta(hours=25)

    def test_hour_during_dst_fold_is_one_real_hour(self):
        # 01:30 happens twice in New York on this date; this is the second one.
        now = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
        start, end = compute_window(_calendar(WindowUnit.HOUR, "America/New_York"), now)
        assert start == datetime(2026, 11, 1, 6, tzinfo=UTC)
        assert end - start == timedelta(hours=1)

    def test_week_starts_monday(self):
        now = datetime(2026, 11, 8, 23, 59, tzinfo=UTC)  # a Sunday
        start, end = compute_window(_calendar(WindowUnit.WEEK), now)
        assert start == datetime(2026, 11, 2, tzinfo=UTC)
        assert start.weekday() == 0
        assert end == datetime(2026, 11, 9, tzinfo=UTC)

    @pytest.mark.parametrize(
        "now,expected_start,expected_end",
        [
            (datetime(2026, 1, 31, 12, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)),
            (datetime(2026, 2, 28, 12, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)),
            (datetime(2028, 2, 29, 12, tzinfo=UTC), datetime(2028, 2, 1, tzinfo=UTC), datetime(2028, 3, 1, tzinfo=UTC)),
            (datetime(2026, 12, 31, 12, tzinfo=UTC), datetime(2026, 12, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)),
        ],
    )
    def test_month_ends(self, now, expected_start, expected_end):
        assert compute_window(_calendar(WindowUnit.MONTH), now) == (expected_start, expected_end)

    def test_month_across_dst_uses_each_dates_offset(self):
        now = datetime(2026, 3, 15, tzinfo=UTC)
        start, end = compute_window(_calendar(WindowUnit.MONTH, "America/New_York"), now)
        assert start == datetime(2026, 3, 1, 5, tzinfo=UTC)  # EST
        assert end == datetime(2026, 4, 1, 4, tzinfo=UTC)  # EDT

    def test_now_is_always_inside_the_window(self):
        spec = _calendar(WindowUnit.DAY, "Asia/Kolkata")
        now = datetime(2026, 6, 1, 18, 29, 59, tzinfo=UTC)
        start, end = compute_window(spec, now)
        assert start <= now < end

    def test_calendar_without_unit_is_an_error(self):
        with pytest.raises(ValueError):
            compute_window(WindowSpec(type=WindowType.CALENDAR), datetime(2026, 1, 1, tzinfo=UTC))


class TestSessionWindows:
    def test_starts_now(self):
        now = datetime(2026, 1, 1, 10, 13, 7, tzinfo=UTC)
        start, end = compute_window(WindowSpec(type=WindowType.SESSION, duration_seconds=10800), now)
        assert start == now
        assert end == datetime(2026, 1, 1, 13, 13, 7, tzinfo=UTC)


class TestSecondsUntil:
    def test_rounds_up_so_clients_never_retry_early(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert seconds_until(now + timedelta(milliseconds=100), now) == 1
        assert seconds_until(now + timedelta(seconds=30), now) == 30

    def test_floors_at_zero(self):
        now = datetime(2026, 1, 1, tzinfo=UTC)
        assert seconds_until(now - timedelta(seconds=5), now) == 0


class TestBucketIdentity:
    SUBJECT = QuotaSubject("u1", "a@example.com", "m1", "claude", "t1")

    def test_shared_bucket_ignores_the_subject(self):
        other = QuotaSubject("u2", "b@example.com", "m2", "gpt", "t2")
        assert bucket_identity((), self.SUBJECT)[0] == bucket_identity((), other)[0]

    def test_dimension_order_does_not_matter(self):
        assert bucket_identity(("user", "model"), self.SUBJECT)[0] == bucket_identity(("model", "user"), self.SUBJECT)[0]

    def test_different_values_get_different_buckets(self):
        other = QuotaSubject("u2", "a@example.com", "m1", "claude", "t1")
        assert bucket_identity(("user",), self.SUBJECT)[0] != bucket_identity(("user",), other)[0]

    def test_requested_name_cannot_spawn_a_bucket(self):
        """A match_pattern lets clients send any model string; only the resolved id counts."""
        renamed = QuotaSubject("u1", "a@example.com", "m1", "claude-custom-suffix", "t1")
        assert bucket_identity(("model",), self.SUBJECT)[0] == bucket_identity(("model",), renamed)[0]

    def test_delimiter_characters_cannot_collide(self):
        a = QuotaSubject('u","model":"x', "e", "m", "n", "t")
        b = QuotaSubject("u", "e", 'x","user":"u', "n", "t")
        assert bucket_identity(("user", "model"), a)[0] != bucket_identity(("user", "model"), b)[0]

    def test_key_fits_the_column(self):
        assert len(bucket_identity(("user",), self.SUBJECT)[0]) == 64

    def test_display_values_are_readable(self):
        _, values = bucket_identity(("user", "model"), self.SUBJECT)
        assert values["user_email"] == "a@example.com"
        assert values["model_name"] == "claude"

    def test_unknown_dimension_is_an_error(self):
        with pytest.raises(ValueError):
            bucket_identity(("x-header",), self.SUBJECT)


class TestEstimates:
    EST = RequestEstimate(
        input_chars=4000, max_tokens=500, input_cost_per_1k=Decimal("0.003"), output_cost_per_1k=Decimal("0.015")
    )

    def test_requests(self):
        assert estimate_amount(Measure.REQUESTS, self.EST) == 1

    def test_tokens(self):
        assert estimate_amount(Measure.TOKENS, self.EST) == 1500

    def test_cost(self):
        assert estimate_amount(Measure.COST, self.EST) == Decimal("0.0105")

    def test_accepts_stored_string_measure(self):
        assert estimate_amount("tokens", self.EST) == 1500

    def test_unknown_measure(self):
        with pytest.raises(ValueError):
            estimate_amount("bananas", self.EST)

    def test_base64_payloads_do_not_inflate_the_estimate(self):
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "A" * 1000}}
        assert "data" not in _strip_base64({"content": [block]})["content"][0]["source"]


class TestActualAmount:
    def test_success_charges_actuals(self):
        usage = ActualUsage(200, tokens=120, cost=Decimal("0.5"))
        assert actual_amount(Measure.REQUESTS, usage, Decimal(1)) == 1
        assert actual_amount(Measure.TOKENS, usage, Decimal(9999)) == 120
        assert actual_amount(Measure.COST, usage, Decimal(9999)) == Decimal("0.5")

    def test_disconnect_is_billable(self):
        usage = ActualUsage(499, tokens=10, cost=Decimal("0.1"))
        assert actual_amount(Measure.REQUESTS, usage, Decimal(1)) == 1
        assert actual_amount(Measure.TOKENS, usage, Decimal(99)) == 10

    def test_provider_error_refunds(self):
        usage = ActualUsage(500)
        assert actual_amount(Measure.REQUESTS, usage, Decimal(1)) == 0
        assert actual_amount(Measure.TOKENS, usage, Decimal(99)) == 0
        assert actual_amount(Measure.COST, usage, Decimal(99)) == 0

    def test_failure_after_paid_usage_still_charges_it(self):
        usage = ActualUsage(500, tokens=40, cost=Decimal("0.2"))
        assert actual_amount(Measure.REQUESTS, usage, Decimal(1)) == 0
        assert actual_amount(Measure.TOKENS, usage, Decimal(99)) == 40
        assert actual_amount(Measure.COST, usage, Decimal(99)) == Decimal("0.2")

    def test_success_with_no_reported_usage_charges_the_estimate(self):
        usage = ActualUsage(200)
        assert usage.usage_missing
        assert actual_amount(Measure.TOKENS, usage, Decimal(99)) == 99
        assert actual_amount(Measure.COST, usage, Decimal("1.5")) == Decimal("1.5")
        assert actual_amount(Measure.REQUESTS, usage, Decimal(1)) == 1


class TestAdmission:
    LIMIT = Decimal(10)

    def test_soft_admits_while_anything_is_left(self):
        assert admits(Enforcement.SOFT, Decimal(9), Decimal(0), Decimal(1000), self.LIMIT)

    def test_soft_blocks_at_the_limit(self):
        assert not admits(Enforcement.SOFT, Decimal(10), Decimal(0), Decimal(1), self.LIMIT)

    def test_soft_counts_in_flight(self):
        assert not admits(Enforcement.SOFT, Decimal(5), Decimal(5), Decimal(1), self.LIMIT)

    def test_strict_requires_the_estimate_to_fit(self):
        assert admits(Enforcement.STRICT, Decimal(5), Decimal(0), Decimal(5), self.LIMIT)
        assert not admits(Enforcement.STRICT, Decimal(5), Decimal(0), Decimal(6), self.LIMIT)
        assert not admits(Enforcement.STRICT, Decimal(5), Decimal(3), Decimal(3), self.LIMIT)

    def test_stored_string_enforcement(self):
        assert not admits("strict", Decimal(0), Decimal(0), Decimal(11), self.LIMIT)
        assert admits("soft", Decimal(0), Decimal(0), Decimal(11), self.LIMIT)


def _denial(name="q", retry_after=60, on_exceeded=None, used="5", limit="5"):
    return QuotaDenial(
        quota_id="id",
        name=name,
        measure="cost",
        limit=Decimal(limit),
        used=Decimal(used),
        in_flight=Decimal(0),
        retry_after=retry_after,
        on_exceeded=on_exceeded if on_exceeded is not None else {},
    )


class TestDenialDecision:
    REROUTE = {"action": "reroute", "target": "haiku"}

    def test_block_beats_reroute(self):
        decision = decide_denial([_denial("a", on_exceeded=self.REROUTE), _denial("b")])
        assert decision.reroute_target is None
        assert decision.primary.name == "b"

    def test_reroute_when_all_ask_for_it_first_by_name(self):
        other = {"action": "reroute", "target": "sonnet"}
        decision = decide_denial([_denial("z", on_exceeded=other), _denial("a", on_exceeded=self.REROUTE)])
        assert decision.reroute_target == "haiku"

    def test_retry_after_is_the_longest_wait(self):
        decision = decide_denial([_denial("a", retry_after=10), _denial("b", retry_after=99)])
        assert decision.retry_after == 99
        assert decision.primary.name == "b"

    def test_retry_after_ignores_quotas_that_waiting_cannot_help(self):
        decision = decide_denial([_denial("a", retry_after=None), _denial("b", retry_after=30)])
        assert decision.retry_after == 30

    def test_no_retry_after_when_waiting_cannot_help(self):
        assert decide_denial([_denial(retry_after=None)]).retry_after is None

    def test_requires_a_denial(self):
        with pytest.raises(ValueError):
            decide_denial([])


class TestDenialResponse:
    def test_status_defaults_to_429(self):
        assert denial_status(_denial()) == 429

    def test_status_override(self):
        assert denial_status(_denial(on_exceeded={"status_code": 402})) == 402

    def test_never_fits_is_a_400_invalid_request(self):
        """SDKs retry 429s automatically, which can't help a request larger than the limit."""
        denial = _denial(retry_after=None)
        assert denial_status(denial) == 400
        assert denial_error_type(denial) == "invalid_request_error"
        assert "can never fit quota 'q'" in denial_message(denial)

    def test_exceeded_is_a_rate_limit_error(self):
        assert denial_error_type(_denial()) == "rate_limit_error"

    def test_explicit_status_wins_even_when_it_never_fits(self):
        denial = _denial(retry_after=None, on_exceeded={"status_code": 429, "message": "custom"})
        assert denial_status(denial) == 429
        assert denial_error_type(denial) == "rate_limit_error"
        assert denial_message(denial) == "custom"

    def test_default_message(self):
        message = denial_message(_denial(name="daily", used="5.5", limit="5"))
        assert message == "Quota 'daily' exceeded (5.5/5 cost)."

    def test_template_variables(self):
        message = denial_message(
            _denial(
                used="7.500000000000000000",
                limit="5.000000000000000000",
                on_exceeded={"message": "{{quota.used}}/{{quota.limit}}, retry in {{quota.retry_after}}s"},
            )
        )
        assert message == "7.5/5, retry in 60s"

    def test_format_amount_has_no_exponent_or_trailing_zeros(self):
        assert format_amount(Decimal("100.000000000000000000")) == "100"
        assert format_amount(Decimal("0E-18")) == "0"
        assert format_amount(Decimal("0.000012000000000000")) == "0.000012"


class TestQuotaDefinition:
    BASE = {"name": "q", "measure": "cost", "limit_value": "5", "window_type": "calendar", "window_unit": "day"}

    def test_defaults(self):
        quota = QuotaDefinition(**self.BASE)
        assert quota.dimensions == ["user"]
        assert quota.enforcement == "soft"
        assert quota.timezone == "UTC"

    def test_shared_quota_has_no_dimensions(self):
        assert QuotaDefinition(**self.BASE, dimensions=[]).dimensions == []

    @pytest.mark.parametrize(
        "change",
        [
            {"limit_value": "0"},
            {"limit_value": "-1"},
            {"measure": "bananas"},
            {"dimensions": ["header"]},
            {"dimensions": ["user", "user"]},
            {"timezone": "Mars/Olympus"},
            {"enforcement": "maybe"},
            {"window_unit": None},
            {"window_unit": "fortnight"},
            {"window_duration_seconds": 60},
            {"window_type": "session"},
            {"on_exceeded": {"action": "reroute"}},
            {"on_exceeded": {"status_code": 200}},
        ],
    )
    def test_invalid(self, change):
        with pytest.raises(ValueError):
            QuotaDefinition(**{**self.BASE, **change})

    def test_session_window(self):
        quota = QuotaDefinition(
            name="q", measure="tokens", limit_value=100, window_type="session", window_duration_seconds=10800
        )
        assert quota.window_duration_seconds == 10800

    def test_session_rejects_unit_and_zero_duration(self):
        base = {"name": "q", "measure": "tokens", "limit_value": 1, "window_type": "session"}
        with pytest.raises(ValueError):
            QuotaDefinition(**base, window_duration_seconds=60, window_unit="day")
        with pytest.raises(ValueError):
            QuotaDefinition(**base, window_duration_seconds=0)
