"""Usage aggregation and cost reporting queries."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import Numeric, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.audit import AuditLog
from ttllm.models.llm_model import LLMModel
from ttllm.models.user import User


def _total_cost_sum():
    """SUM of the stored authoritative ``total_cost`` (a String column), cast to numeric.

    ``total_cost`` is written as ``str(Decimal)``. NULLs (e.g. error rows) are skipped by
    SUM; empty strings would fail the cast, so they are mapped to NULL first.
    """
    return func.sum(cast(func.nullif(AuditLog.total_cost, ""), Numeric(38, 18)))


_WINDOW_MEASURES = {
    "cost": lambda: _total_cost_sum(),
    "tokens": lambda: func.sum(AuditLog.input_tokens + AuditLog.output_tokens),
    "requests": lambda: func.count(AuditLog.id),
}

_BILLABLE_STATUS_CODES = (200, 499)


async def get_window_aggregate(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    measure: str,
    window_seconds: int,
    per: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate a user's successful usage over a trailing moving window.

    Used by the rules engine's quota conditions. ``measure`` is one of
    ``cost`` / ``tokens`` / ``requests``. Returns ``{"value", "oldest_ts"}``
    where ``oldest_ts`` is the earliest contributing row (used to compute when
    the window frees up). Only billable rows count — completed requests and
    client-disconnected streams that still reached final provider usage metadata.

    ``per`` optionally narrows the window by dimension; currently only
    ``{"model": "<name>"}`` is supported (exact model-name scoping).
    """
    measure_fn = _WINDOW_MEASURES.get(measure)
    if measure_fn is None:
        raise ValueError(f"unknown quota measure: {measure}")

    now = now or datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=window_seconds)

    query = select(
        measure_fn().label("value"),
        func.min(AuditLog.created_at).label("oldest_ts"),
    ).where(
        AuditLog.user_id == user_id,
        AuditLog.created_at >= window_start,
        AuditLog.status_code.in_(_BILLABLE_STATUS_CODES),
    )

    per = per or {}
    model_name = per.get("model")
    if model_name is not None:
        query = query.join(LLMModel, AuditLog.model_id == LLMModel.id).where(
            LLMModel.name == model_name
        )

    row = (await db.execute(query)).one()
    value = row.value
    if measure == "requests":
        value = int(value or 0)
    else:
        value = Decimal(value) if value is not None else Decimal("0")
    return {"value": value, "oldest_ts": row.oldest_ts}


async def get_usage_summary(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Get aggregated usage statistics."""
    query = select(
        func.count(AuditLog.id).label("total_requests"),
        func.sum(AuditLog.input_tokens).label("total_input_tokens"),
        func.sum(AuditLog.output_tokens).label("total_output_tokens"),
        func.avg(AuditLog.latency_ms).label("avg_latency_ms"),
        _total_cost_sum().label("total_cost"),
    )

    if user_id:
        query = query.where(AuditLog.user_id == user_id)
    if model_id:
        query = query.where(AuditLog.model_id == model_id)
    if since:
        query = query.where(AuditLog.created_at >= since)
    if until:
        query = query.where(AuditLog.created_at <= until)

    result = await db.execute(query)
    row = result.one()

    return {
        "total_requests": row.total_requests or 0,
        "total_input_tokens": row.total_input_tokens or 0,
        "total_output_tokens": row.total_output_tokens or 0,
        "avg_latency_ms": round(float(row.avg_latency_ms or 0), 1),
        "total_cost": str(row.total_cost or 0),
    }


async def get_cost_breakdown(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[dict[str, Any]]:
    """Get cost breakdown grouped by model.

    Sums the authoritative ``total_cost`` stored on each audit row (computed by the provider,
    so it already accounts for cache and any provider-specific cost dimensions) rather than
    recomputing from token counts.
    """
    query = (
        select(
            LLMModel.name.label("model_name"),
            func.count(AuditLog.id).label("request_count"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            _total_cost_sum().label("total_cost"),
        )
        .join(LLMModel, AuditLog.model_id == LLMModel.id)
        .group_by(LLMModel.name)
    )

    if user_id:
        query = query.where(AuditLog.user_id == user_id)
    if model_id:
        query = query.where(AuditLog.model_id == model_id)
    if since:
        query = query.where(AuditLog.created_at >= since)
    if until:
        query = query.where(AuditLog.created_at <= until)

    result = await db.execute(query)
    breakdown = []
    for row in result.all():
        breakdown.append({
            "model_name": row.model_name,
            "request_count": row.request_count,
            "input_tokens": row.input_tokens or 0,
            "output_tokens": row.output_tokens or 0,
            "total_cost": str(row.total_cost or 0),
        })

    return breakdown


async def get_user_usage_summary(
    db: AsyncSession,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Get usage summary grouped by user, ordered by total cost descending.

    Highest-spending users come first, so passing ``limit`` yields the top N users by cost.
    """
    total_cost = _total_cost_sum()
    query = (
        select(
            User.id.label("user_id"),
            User.name.label("user_name"),
            User.email.label("user_email"),
            func.count(AuditLog.id).label("request_count"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            total_cost.label("total_cost"),
        )
        .join(User, AuditLog.user_id == User.id)
        .group_by(User.id, User.name, User.email)
        .order_by(func.coalesce(total_cost, 0).desc())
    )

    if since:
        query = query.where(AuditLog.created_at >= since)
    if until:
        query = query.where(AuditLog.created_at <= until)
    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    return [
        {
            "user_id": str(row.user_id),
            "user_name": row.user_name,
            "user_email": row.user_email,
            "request_count": row.request_count,
            "input_tokens": row.input_tokens or 0,
            "output_tokens": row.output_tokens or 0,
            "total_cost": str(row.total_cost or 0),
        }
        for row in result.all()
    ]
