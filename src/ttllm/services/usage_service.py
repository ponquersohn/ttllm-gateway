"""Usage aggregation and cost reporting queries."""

from __future__ import annotations

import uuid
from datetime import datetime
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
) -> list[dict[str, Any]]:
    """Get usage summary grouped by user."""
    query = (
        select(
            User.id.label("user_id"),
            User.name.label("user_name"),
            User.email.label("user_email"),
            func.count(AuditLog.id).label("request_count"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            _total_cost_sum().label("total_cost"),
        )
        .join(User, AuditLog.user_id == User.id)
        .group_by(User.id, User.name, User.email)
    )

    if since:
        query = query.where(AuditLog.created_at >= since)
    if until:
        query = query.where(AuditLog.created_at <= until)

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
