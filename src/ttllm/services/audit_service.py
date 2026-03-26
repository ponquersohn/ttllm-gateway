"""Audit log writing."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.config import settings
from ttllm.models.audit import AuditLog, AuditLogBody


async def log_request(
    db: AsyncSession,
    user_id: uuid.UUID,
    model_id: uuid.UUID,
    request_id: uuid.UUID,
    input_tokens: int,
    output_tokens: int,
    total_cost: str | None = None,
    latency_ms: int = 0,
    status_code: int = 200,
    error_message: str | None = None,
    metadata_json: dict | None = None,
    request_body: dict | None = None,
    response_body: dict | None = None,
    log_bodies: bool | None = None,
) -> AuditLog:
    """Write an audit log entry. Optionally includes request/response bodies."""
    audit = AuditLog(
        user_id=user_id,
        model_id=model_id,
        request_id=request_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost=total_cost,
        latency_ms=latency_ms,
        status_code=status_code,
        error_message=error_message,
        metadata_json=metadata_json,
    )
    db.add(audit)
    await db.flush()

    # Write bodies if enabled globally or per-request
    should_log_bodies = log_bodies if log_bodies is not None else settings.engine.log_request_bodies
    if should_log_bodies and (request_body or response_body):
        body = AuditLogBody(
            audit_log_id=audit.id,
            request_body=request_body,
            response_body=response_body,
        )
        db.add(body)

    await db.commit()
    await db.refresh(audit)
    return audit


async def get_audit_logs(
    db: AsyncSession,
    user_id: uuid.UUID | None = None,
    model_id: uuid.UUID | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[AuditLog], int]:
    """Query audit logs with optional filters."""
    query = select(AuditLog)
    count_query = select(AuditLog.id)

    if user_id:
        query = query.where(AuditLog.user_id == user_id)
        count_query = count_query.where(AuditLog.user_id == user_id)
    if model_id:
        query = query.where(AuditLog.model_id == model_id)
        count_query = count_query.where(AuditLog.model_id == model_id)
    if since:
        query = query.where(AuditLog.created_at >= since)
        count_query = count_query.where(AuditLog.created_at >= since)
    if until:
        query = query.where(AuditLog.created_at <= until)
        count_query = count_query.where(AuditLog.created_at <= until)

    count_result = await db.execute(count_query)
    total = len(count_result.all())

    query = query.offset(offset).limit(limit).order_by(AuditLog.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def get_audit_log_body(
    db: AsyncSession,
    audit_log_id: uuid.UUID,
) -> AuditLogBody | None:
    result = await db.execute(
        select(AuditLogBody).where(AuditLogBody.audit_log_id == audit_log_id)
    )
    return result.scalar_one_or_none()
