"""Pydantic schemas for the admin API."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

_SENSITIVE_KEY_PATTERN = re.compile(r"(secret|key|password|token|credential)", re.IGNORECASE)
_REDACTED = "***REDACTED***"


def _redact_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values whose keys look sensitive."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _redact_dict(v)
        elif _SENSITIVE_KEY_PATTERN.search(k):
            out[k] = _REDACTED
        elif isinstance(v, str) and v.startswith("secret://"):
            out[k] = _REDACTED
        else:
            out[k] = v
    return out


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


def _validate_match_pattern(v: str | None) -> str | None:
    if v is None:
        return None
    if len(v) > 512:
        raise ValueError("match_pattern must be at most 512 characters")
    try:
        re.compile(v)
    except re.error as e:
        raise ValueError(f"match_pattern is not a valid regex: {e}")
    return v


class ModelCreate(BaseModel):
    name: str
    provider: str
    provider_model_id: str
    config_json: dict[str, Any] = {}
    input_cost_per_1k: Decimal = Decimal("0")
    output_cost_per_1k: Decimal = Decimal("0")
    cache_read_cost_per_1k: Decimal = Decimal("0")
    cache_write_cost_per_1k: Decimal = Decimal("0")
    match_pattern: str | None = None

    @field_validator("match_pattern")
    @classmethod
    def check_match_pattern(cls, v: str | None) -> str | None:
        return _validate_match_pattern(v)


class ModelUpdate(BaseModel):
    name: str | None = None
    provider: str | None = None
    provider_model_id: str | None = None
    config_json: dict[str, Any] | None = None
    merge_config: bool = False
    input_cost_per_1k: Decimal | None = None
    output_cost_per_1k: Decimal | None = None
    cache_read_cost_per_1k: Decimal | None = None
    cache_write_cost_per_1k: Decimal | None = None
    is_active: bool | None = None
    match_pattern: str | None = None

    @field_validator("match_pattern")
    @classmethod
    def check_match_pattern(cls, v: str | None) -> str | None:
        return _validate_match_pattern(v)


class ModelResponse(BaseModel):
    id: uuid.UUID
    name: str
    provider: str
    provider_model_id: str
    config_json: dict[str, Any]
    input_cost_per_1k: Decimal
    output_cost_per_1k: Decimal
    cache_read_cost_per_1k: Decimal
    cache_write_cost_per_1k: Decimal
    match_pattern: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def redact_config_secrets(self):
        if self.config_json:
            self.config_json = _redact_dict(self.config_json)
        return self


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
    provider_metadata: dict | None = None
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
    total_cost: str = "0"


class CostBreakdownItem(BaseModel):
    model_name: str
    request_count: int
    input_tokens: int
    output_tokens: int
    total_cost: str




# --- Rules ---


class RuleCreate(BaseModel):
    name: str
    description: str | None = None
    weight: int = 0
    enabled: bool = True
    conditions: dict[str, Any]
    action: dict[str, Any]

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: dict[str, Any]) -> dict[str, Any]:
        from ttllm.schemas.rules import validate_condition_group_dict
        return validate_condition_group_dict(v)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: dict[str, Any]) -> dict[str, Any]:
        from ttllm.schemas.rules import validate_action_dict
        return validate_action_dict(v)


class RuleUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    weight: int | None = None
    enabled: bool | None = None
    conditions: dict[str, Any] | None = None
    action: dict[str, Any] | None = None

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return None
        from ttllm.schemas.rules import validate_condition_group_dict
        return validate_condition_group_dict(v)

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return None
        from ttllm.schemas.rules import validate_action_dict
        return validate_action_dict(v)


class RuleResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    weight: int
    enabled: bool
    conditions: dict[str, Any]
    action: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


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


class StatusCheck(BaseModel):
    name: str
    status: str  # "ok" | "warning" | "error"
    message: str | None = None


class ServerStatusResponse(BaseModel):
    version: str
    status: str  # "ok" | "degraded"
    checks: list[StatusCheck] = []
