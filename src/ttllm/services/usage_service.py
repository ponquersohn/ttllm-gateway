"""Usage aggregation and cost reporting queries."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Numeric, String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.audit import AuditLog


def _total_cost_sum():
    """SUM of the stored authoritative ``total_cost`` (a String column), cast to numeric.

    ``total_cost`` is written as ``str(Decimal)``. NULLs (e.g. error rows) are skipped by
    SUM; empty strings would fail the cast, so they are mapped to NULL first.
    """
    return func.sum(cast(func.nullif(AuditLog.total_cost, ""), Numeric(38, 18)))


def _quota_blocked():
    """Audit rows written for a quota denial (they carry ``metadata_json["quota"]``).

    They record a request that never reached a provider, so usage aggregates leave them out:
    a client retrying a blocked request in a loop would otherwise inflate request counts and
    drag down average latency. The summary reports them separately.
    """
    return func.coalesce(AuditLog.metadata_json.has_key("quota"), False)


def _served():
    return _quota_blocked().is_(False)


async def get_usage_summary(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Get aggregated usage statistics."""
    query = select(
        func.count(AuditLog.id).filter(_served()).label("total_requests"),
        func.count(AuditLog.id).filter(_quota_blocked()).label("quota_blocked_requests"),
        func.sum(AuditLog.input_tokens).label("total_input_tokens"),
        func.sum(AuditLog.output_tokens).label("total_output_tokens"),
        func.avg(AuditLog.latency_ms).filter(_served()).label("avg_latency_ms"),
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
        "quota_blocked_requests": row.quota_blocked_requests or 0,
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
            AuditLog.model_name.label("model_name"),
            func.count(AuditLog.id).label("request_count"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            _total_cost_sum().label("total_cost"),
        )
        .where(_served())
        .group_by(AuditLog.model_name)
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
    Grouped by ``user_email`` (the user's immutable identity, not joined against the live
    ``users`` row) rather than ``user_id``, mirroring ``get_cost_breakdown``'s grouping by
    ``model_name`` -- a deleted user's history isn't dropped, and email can't drift
    underneath the report. ``user_id``/``user_name`` are display columns only (``name`` is
    still mutable, so ``func.max`` picks an arbitrary one if it actually changed).
    """
    total_cost = _total_cost_sum()
    user_name = AuditLog.user_snapshot["name"].astext
    query = (
        select(
            # Postgres has no MAX() aggregate for uuid (only comparison operators, no
            # built-in aggregate function) -- cast to text first, same as user_id's
            # eventual str() in the response.
            func.max(cast(AuditLog.user_id, String)).label("user_id"),
            func.max(user_name).label("user_name"),
            AuditLog.user_email.label("user_email"),
            func.count(AuditLog.id).label("request_count"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            total_cost.label("total_cost"),
        )
        .where(_served())
        .group_by(AuditLog.user_email)
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
