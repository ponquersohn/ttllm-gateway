from datetime import datetime

from ttllm.cli import usage


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 6, 18, 12, 30, tzinfo=tz)


def test_normalize_time_leaves_iso_values_unchanged():
    assert usage._normalize_time("2026-06-17T00:00:00+02:00") == "2026-06-17T00:00:00+02:00"


def test_normalize_time_converts_relative_hours(monkeypatch):
    monkeypatch.setattr(usage, "datetime", FixedDateTime)

    assert usage._normalize_time("-24h") == "2026-06-17T12:30:00+00:00"


def test_normalize_time_converts_relative_days(monkeypatch):
    monkeypatch.setattr(usage, "datetime", FixedDateTime)

    assert usage._normalize_time("-7d") == "2026-06-11T12:30:00+00:00"
