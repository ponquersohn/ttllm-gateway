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
