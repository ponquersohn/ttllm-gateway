"""Test fixtures for TTLLM tests."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ttllm.api.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.name = "Test User"
    user.email = "test@example.com"
    user.is_active = True
    return user


@pytest.fixture
def mock_admin_user():
    user = MagicMock()
    user.id = uuid.uuid4()
    user.name = "Admin"
    user.email = "admin@example.com"
    user.is_active = True
    return user


@pytest.fixture
def mock_llm_model():
    from decimal import Decimal

    model = MagicMock()
    model.id = uuid.uuid4()
    model.name = "claude-3-sonnet"
    model.provider = "bedrock"
    model.provider_model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    model.config_json = {}
    model.input_cost_per_1k = Decimal("0.003")
    model.output_cost_per_1k = Decimal("0.015")
    model.is_active = True
    return model
