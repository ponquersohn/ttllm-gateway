"""Pydantic schemas for the admin API."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel


# --- Users ---


class WhoamiResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    groups: list[str] = []
    effective_permissions: list[str]
    available_permissions: list[str]


class UserCreate(BaseModel):
    name: str
    email: str
    password: str | None = None


class UserUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    password: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    identity_provider: str | None
    groups: list[str] = []
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Models ---


class ModelCreate(BaseModel):
    name: str
    provider: str
    provider_model_id: str
    config_json: dict[str, Any] = {}
    input_cost_per_1k: Decimal = Decimal("0")
    output_cost_per_1k: Decimal = Decimal("0")


class ModelUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    provider_model_id: str | None = None
    config_json: dict[str, Any] | None = None
    input_cost_per_1k: Decimal | None = None
    output_cost_per_1k: Decimal | None = None
    is_active: bool | None = None


class ModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    provider_model_id: str
    config_json: dict[str, Any]
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Assignments ---


class AssignRequest(BaseModel):
    user_ids: list[uuid.UUID]


class GroupAssignRequest(BaseModel):
    group_ids: list[uuid.UUID]


# --- Audit ---


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    model_id: uuid.UUID
    request_id: uuid.UUID
    input_tokens: int
    output_tokens: int
    total_cost: str | None
    latency_ms: int
    status_code: int
    error_message: str | None
    metadata_json: dict | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogBodyResponse(BaseModel):
    request_body: dict | None
    response_body: dict | None

    model_config = {"from_attributes": True}


# --- Usage ---


class UsageSummaryResponse(BaseModel):
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    avg_latency_ms: float


class CostBreakdownItem(BaseModel):
    model_name: str
    request_count: int
    input_tokens: int
    output_tokens: int
    total_cost: str


# --- Secrets ---


class SecretCreate(BaseModel):
    name: str
    value: str
    description: str | None = None


class SecretUpdate(BaseModel):
    value: str | None = None
    description: str | None = None


class SecretResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Server ---


class ServerStatusResponse(BaseModel):
    version: str
    status: str
