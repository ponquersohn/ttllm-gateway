"""Pydantic schemas for the /me self-service API."""

from __future__ import annotations

import uuid

from pydantic import BaseModel


class SelfTokenCreate(BaseModel):
    label: str | None = None
    ttl_days: int | None = None
    permissions: list[str] | None = None  # default: ["llm.invoke"]


class AvailableModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    match_pattern: str | None = None

    model_config = {"from_attributes": True}
