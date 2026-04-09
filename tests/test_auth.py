"""Tests for JWT and permission utilities."""

import uuid
from datetime import timedelta

from ttllm.core.jwt import JWTConfig, create_access_token, decode_token
from ttllm.core.permissions import Permissions, PermissionRegistry, PermissionDef


class TestJWT:
    _config = JWTConfig(secret_key="test-secret")

    def test_roundtrip(self):
        user_id = uuid.uuid4()
        jti = uuid.uuid4()
        token = create_access_token(
            user_id=user_id,
            permissions=["llm.invoke"],
            jti=jti,
            ttl=timedelta(minutes=5),
            config=self._config,
        )
        payload = decode_token(token, self._config)
        assert payload.sub == user_id
        assert payload.jti == jti
        assert payload.permissions == ["llm.invoke"]

    def test_permissions_sorted(self):
        token = create_access_token(
            user_id=uuid.uuid4(),
            permissions=["model.view", "audit.view", "model.assign"],
            jti=uuid.uuid4(),
            ttl=timedelta(minutes=5),
            config=self._config,
        )
        payload = decode_token(token, self._config)
        assert payload.permissions == ["audit.view", "model.assign", "model.view"]


class TestPermissionRegistry:
    def _make_registry(self) -> PermissionRegistry:
        reg = PermissionRegistry()
        reg.permissions["llm.invoke"] = PermissionDef(
            name="llm.invoke", description="", category="gateway"
        )
        reg.permissions["model.view"] = PermissionDef(
            name="model.view", description="", category="management"
        )
        reg.permissions["model.assign"] = PermissionDef(
            name="model.assign", description="", category="management"
        )
        return reg

    def test_filter_by_category(self):
        reg = self._make_registry()
        all_perms = {"llm.invoke", "model.view", "model.assign"}
        assert reg.filter_by_category(all_perms, "gateway") == {"llm.invoke"}
        assert reg.filter_by_category(all_perms, "management") == {"model.view", "model.assign"}

    def test_check_has_permission(self):
        reg = self._make_registry()
        assert reg.check({"model.view"}, "model.view") is True

    def test_check_missing_permission(self):
        reg = self._make_registry()
        assert reg.check({"model.view"}, "model.assign") is False

    def test_check_unknown_permission(self):
        reg = self._make_registry()
        assert reg.check({"foo.bar"}, "foo.bar") is False

    def test_validate_permission(self):
        reg = self._make_registry()
        assert reg.validate_permission("llm.invoke") is True
        assert reg.validate_permission("model.view") is True
        assert reg.validate_permission("nonexistent") is False


class TestPermissionsClass:
    def test_constants_match_dotted_names(self):
        assert Permissions.USER_VIEW == "user.view"
        assert Permissions.USER_CREATE == "user.create"
        assert Permissions.MODEL_ASSIGN == "model.assign"
        assert Permissions.LLM_INVOKE == "llm.invoke"

    def test_registry_has_all_permissions(self):
        registry = Permissions.get_registry()
        assert len(registry.permissions) == 22

    def test_registry_contains_all_constants(self):
        registry = Permissions.get_registry()
        assert "user.view" in registry.permissions
        assert "llm.invoke" in registry.permissions
        assert "token.create" in registry.permissions
        assert "audit.view" in registry.permissions

    def test_registry_validates_permissions(self):
        registry = Permissions.get_registry()
        assert registry.validate_permission(Permissions.USER_VIEW) is True
        assert registry.validate_permission("nonexistent") is False
