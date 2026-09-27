"""Unit tests for display_name wiring in model_service.

These exercise create_model/update_model against a mocked AsyncSession (no real DB
needed) — the point is to pin down that `display_name` is actually persisted and
mutable, since both create_model() and update_model()'s _MUTABLE_FIELDS previously
dropped it silently.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ttllm.models.llm_model import LLMModel
from ttllm.services import model_service


@pytest.mark.asyncio
async def test_create_model_persists_display_name():
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    model = await model_service.create_model(
        db,
        name="claude-sonnet",
        provider="bedrock",
        provider_model_id="anthropic.claude-sonnet-4-20250514-v1:0",
        display_name="TTLLM - Claude Sonnet",
    )

    assert model.name == "claude-sonnet"
    assert model.display_name == "TTLLM - Claude Sonnet"
    db.add.assert_called_once_with(model)
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_model_display_name_defaults_to_none():
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    model = await model_service.create_model(
        db, name="claude-sonnet", provider="bedrock", provider_model_id="m"
    )

    assert model.display_name is None


@pytest.mark.asyncio
async def test_update_model_sets_display_name():
    existing = LLMModel(
        id=uuid.uuid4(),
        name="claude-sonnet",
        display_name=None,
        provider="bedrock",
        provider_model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    )

    db = MagicMock()
    db.get = AsyncMock(return_value=existing)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    updated = await model_service.update_model(
        db, existing.id, display_name="TTLLM - Claude Sonnet"
    )

    assert updated is existing
    assert updated.display_name == "TTLLM - Claude Sonnet"


@pytest.mark.asyncio
async def test_update_model_can_clear_display_name_with_empty_string():
    existing = LLMModel(
        id=uuid.uuid4(),
        name="claude-sonnet",
        display_name="TTLLM - Claude Sonnet",
        provider="bedrock",
        provider_model_id="m",
    )

    db = MagicMock()
    db.get = AsyncMock(return_value=existing)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    updated = await model_service.update_model(db, existing.id, display_name="")

    assert updated.display_name == ""
