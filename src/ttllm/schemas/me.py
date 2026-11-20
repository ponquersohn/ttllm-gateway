"""Pydantic schemas for the /me self-service API."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class SelfTokenCreate(BaseModel):
    label: str | None = None
    ttl_days: int | None = None
    permissions: list[str] | None = None  # default: ["llm.invoke"]


class AvailableModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    display_name: str | None = None
    provider: str
    match_pattern: str | None = None

    model_config = {"from_attributes": True}


class QuotaOverrideCode(BaseModel):
    code: str = Field(min_length=1, max_length=256)


class QuotaOverridePreview(BaseModel):
    """What a code grants, shown before it's redeemed."""

    reason: str
    created_by_email: str | None
    override_duration_seconds: int
    code_expires_at: datetime


class MyQuotaOverride(BaseModel):
    id: uuid.UUID
    reason: str
    created_by_email: str | None
    redeemed_at: datetime
    override_expires_at: datetime

    model_config = {"from_attributes": True}
