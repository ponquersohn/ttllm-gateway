"""Pydantic v2 schemas for quota limits and counters."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ttllm.models.quota import LimitScope, WindowKind


class TokenLimitCreate(BaseModel):
    scope: LimitScope
    user_id: uuid.UUID | None = None
    group_id: uuid.UUID | None = None
    window_kind: WindowKind
    token_cap: int = Field(..., gt=0, description="Maximum tokens allowed in this window.")


class TokenLimitUpdate(BaseModel):
    token_cap: int = Field(..., gt=0, description="New maximum tokens allowed in this window.")


class TokenLimitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    scope: LimitScope
    user_id: uuid.UUID | None
    group_id: uuid.UUID | None
    window_kind: WindowKind
    token_cap: int
    created_at: datetime
    updated_at: datetime


class UsageCounterResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: uuid.UUID
    window_kind: WindowKind
    window_start: datetime
    tokens_used: int
    cooldown_until: datetime | None
