"""Tests for the provider registry."""

from unittest.mock import MagicMock

import pytest

from ttllm.core.provider import ProviderRegistry


class TestProviderRegistry:
    def test_register_and_get(self):
        reg = ProviderRegistry()
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.provider = "test"

        fake_chat_model = MagicMock()
        reg.register("test", lambda m, p: fake_chat_model)

        result = reg.get_chat_model(mock_model, {"max_tokens": 100})
        assert result is fake_chat_model

    def test_cache_hit(self):
        reg = ProviderRegistry()
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.provider = "test"

        call_count = 0

        def factory(m, p):
            nonlocal call_count
            call_count += 1
            return MagicMock()

        reg.register("test", factory)
        params = {"max_tokens": 100}

        r1 = reg.get_chat_model(mock_model, params)
        r2 = reg.get_chat_model(mock_model, params)
        assert r1 is r2
        assert call_count == 1

    def test_unknown_provider(self):
        reg = ProviderRegistry()
        mock_model = MagicMock()
        mock_model.id = "test-id"
        mock_model.provider = "unknown"

        with pytest.raises(ValueError, match="Unknown provider"):
            reg.get_chat_model(mock_model, {})

    def test_lru_eviction(self):
        reg = ProviderRegistry(max_cache_size=2)
        reg.register("test", lambda m, p: MagicMock())

        for i in range(3):
            mock = MagicMock()
            mock.id = f"model-{i}"
            mock.provider = "test"
            reg.get_chat_model(mock, {"max_tokens": 100})

        assert len(reg._cache) == 2

    def test_supported_providers(self):
        reg = ProviderRegistry()
        reg.register("bedrock", lambda m, p: MagicMock())
        reg.register("openai", lambda m, p: MagicMock())
        assert set(reg.supported_providers) == {"bedrock", "openai"}
