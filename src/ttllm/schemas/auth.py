"""Pydantic schemas for authentication, groups, and tokens."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


# --- Auth ---


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginTokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


# --- Device code flow (RFC 8628) ---


class DeviceCodeRequest(BaseModel):
    client_name: str | None = None


class DeviceCodeResponse(BaseModel):
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


class DeviceTokenRequest(BaseModel):
    device_code: str


class DeviceLookupResponse(BaseModel):
    client_name: str | None
    requester_ip: str | None
    requester_user_agent: str | None
    created_at: datetime
    expires_at: datetime


class DeviceApproveRequest(BaseModel):
    user_code: str
    approve: bool = True


# --- Groups ---


class GroupCreate(BaseModel):
    name: str
    description: str | None = None


class GroupUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class GroupResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    permissions: list[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class GroupPermissionAssign(BaseModel):
    permission: str


class UserPermissionAssign(BaseModel):
    permissions: list[str]


class GroupMemberAssign(BaseModel):
    user_ids: list[uuid.UUID]


# --- Tokens ---


class TokenCreate(BaseModel):
    user_id: uuid.UUID | None = None
    label: str | None = None
    ttl_days: int | None = None
    permissions: list[str] | None = None  # default: ["llm.invoke"]


class TokenResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    user_email: str | None = None
    label: str | None
    permissions: list[str]
    is_active: bool
    created_at: datetime
    expires_at: datetime | None

    model_config = {"from_attributes": True}


class TokenCreatedResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    permissions: list[str]
    id: uuid.UUID
    label: str | None
    expires_at: datetime | None
