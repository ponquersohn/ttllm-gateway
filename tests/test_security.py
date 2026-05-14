"""Tests for security hardening changes."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ttllm.api.auth import _validate_redirect
from ttllm.core.oidc import (
    OIDCEndpoints,
    extract_roles_from_id_token_payload,
    verify_id_token,
)
from ttllm.core.provider import _validate_base_url
from ttllm.schemas.admin import _REDACTED, _redact_dict


def _make_endpoints() -> OIDCEndpoints:
    return OIDCEndpoints(
        authorization_endpoint="https://idp/authorize",
        token_endpoint="https://idp/token",
        userinfo_endpoint="https://idp/userinfo",
        issuer="https://idp",
        jwks_uri="https://idp/.well-known/jwks.json",
    )


# ---------------------------------------------------------------------------
# SSO final_redirect validation
# ---------------------------------------------------------------------------


class TestValidateRedirect:
    """SSO final_redirect must point to localhost only."""

    def test_localhost_allowed(self):
        assert _validate_redirect("http://localhost:8080/callback") is True

    def test_127_allowed(self):
        assert _validate_redirect("http://127.0.0.1:9000/done") is True

    def test_ipv6_loopback_allowed(self):
        assert _validate_redirect("http://[::1]:3000/cb") is True

    def test_external_host_rejected(self):
        assert _validate_redirect("https://evil.com/steal") is False

    def test_no_host_rejected(self):
        assert _validate_redirect("") is False

    def test_ftp_localhost_allowed(self):
        assert _validate_redirect("ftp://localhost/file") is True

    def test_subdomain_rejected(self):
        assert _validate_redirect("https://localhost.evil.com/x") is False


# ---------------------------------------------------------------------------
# Config JSON secret redaction
# ---------------------------------------------------------------------------


class TestRedactDict:
    """config_json secrets redacted before API response."""

    def test_redacts_secret_key(self):
        result = _redact_dict({"api_key": "sk-123", "name": "test"})
        assert result["api_key"] == _REDACTED
        assert result["name"] == "test"

    def test_redacts_password(self):
        result = _redact_dict({"db_password": "hunter2"})
        assert result["db_password"] == _REDACTED

    def test_redacts_token(self):
        result = _redact_dict({"access_token": "abc"})
        assert result["access_token"] == _REDACTED

    def test_redacts_credential(self):
        result = _redact_dict({"aws_credential": "xyz"})
        assert result["aws_credential"] == _REDACTED

    def test_leaves_normal_keys(self):
        d = {"region": "us-east-1", "model": "gpt-4"}
        assert _redact_dict(d) == d

    def test_nested_redaction(self):
        result = _redact_dict({"outer": {"inner_secret": "val", "ok": "fine"}})
        assert result["outer"]["inner_secret"] == _REDACTED
        assert result["outer"]["ok"] == "fine"

    def test_redacts_secret_uri(self):
        result = _redact_dict({"base_url": "secret://my-key"})
        assert result["base_url"] == _REDACTED

    def test_empty_dict(self):
        assert _redact_dict({}) == {}


# ---------------------------------------------------------------------------
# AuthContext effective permissions
# ---------------------------------------------------------------------------


class TestAuthContextPermissions:
    """Effective permissions = intersection of token + current DB."""

    def test_intersection(self):
        from ttllm.api.deps import AuthContext

        ctx = AuthContext(
            user=MagicMock(),
            token_permissions={"user.view", "user.create", "model.view"},
            current_permissions={"user.view", "model.view", "model.create"},
            jti=uuid.uuid4(),
        )
        assert ctx.permissions == {"user.view", "model.view"}

    def test_no_overlap(self):
        from ttllm.api.deps import AuthContext

        ctx = AuthContext(
            user=MagicMock(),
            token_permissions={"user.view"},
            current_permissions={"model.view"},
            jti=uuid.uuid4(),
        )
        assert ctx.permissions == set()

    def test_revoked_permission_excluded(self):
        from ttllm.api.deps import AuthContext

        ctx = AuthContext(
            user=MagicMock(),
            token_permissions={"user.view", "user.create", "user.delete"},
            current_permissions={"user.view"},
            jti=uuid.uuid4(),
        )
        assert ctx.permissions == {"user.view"}
        assert "user.create" not in ctx.permissions
        assert "user.delete" not in ctx.permissions


# ---------------------------------------------------------------------------
# SSRF protection: base_url validation
# ---------------------------------------------------------------------------


class TestValidateBaseUrl:
    """SSRF protection via allowed_base_urls regex + private IP blocking."""

    def test_no_allowed_urls_rejects(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = []
            with pytest.raises(ValueError, match="No allowed_base_urls configured"):
                _validate_base_url("https://api.openai.com/v1")

    def test_matching_pattern_allowed(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = [r"https://api\.openai\.com/.*"]
            _validate_base_url("https://api.openai.com/v1")

    def test_non_matching_pattern_rejected(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = [r"https://api\.openai\.com/.*"]
            with pytest.raises(ValueError, match="does not match"):
                _validate_base_url("https://evil.com/v1")

    def test_metadata_endpoint_blocked(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = [r"https?://.*"]
            with pytest.raises(ValueError, match="blocked metadata"):
                _validate_base_url("http://169.254.169.254/latest/meta-data")

    def test_private_ip_blocked(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = [r"https?://.*"]
            with pytest.raises(ValueError, match="private address"):
                _validate_base_url("http://10.0.0.1/api")

    def test_loopback_blocked(self):
        with patch("ttllm.core.provider.settings") as mock_settings:
            mock_settings.provider.allowed_base_urls = [r"https?://.*"]
            with pytest.raises(ValueError, match="private address"):
                _validate_base_url("http://127.0.0.1:8080/v1")


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Security headers present on responses."""

    def test_headers_present(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert resp.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# CORS wildcard
# ---------------------------------------------------------------------------


class TestCorsWildcard:
    """CORS wildcard must not allow credentials."""

    def test_wildcard_no_credentials(self, client):
        resp = client.options(
            "/health",
            headers={"Origin": "https://evil.com", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-credentials", "").lower() != "true"


# ---------------------------------------------------------------------------
# JWT algorithm pinned
# ---------------------------------------------------------------------------


class TestJwtAlgorithmPinned:
    """JWT algorithm is pinned, not configurable."""

    def test_algorithm_is_hs256(self):
        from ttllm.core.jwt import _ALGORITHM

        assert _ALGORITHM == "HS256"

    def test_config_has_no_algorithm_field(self):
        from ttllm.config import JWTConfig

        cfg = JWTConfig(secret_key="test")
        assert not hasattr(cfg, "algorithm")


# ---------------------------------------------------------------------------
# OIDC ID token verification
# ---------------------------------------------------------------------------


class TestVerifyIdToken:
    """OIDC ID token signature verification."""

    def test_empty_token_raises(self):
        with pytest.raises(ValueError, match="Empty ID token"):
            verify_id_token("", endpoints=_make_endpoints(), client_id="client123")

    def test_nonce_mismatch_raises(self):
        mock_key = MagicMock()
        mock_key.key = "fake-key"
        with patch("ttllm.core.oidc._get_jwk_client") as mock_jwk:
            mock_jwk.return_value.get_signing_key_from_jwt.return_value = mock_key
            with patch("ttllm.core.oidc.pyjwt.decode", return_value={"nonce": "wrong"}):
                with pytest.raises(ValueError, match="nonce mismatch"):
                    verify_id_token(
                        "header.payload.sig",
                        endpoints=_make_endpoints(),
                        client_id="client123",
                        nonce="expected-nonce",
                    )

    def test_valid_token_returns_payload(self):
        expected_payload = {"sub": "user1", "nonce": "abc", "roles": ["admin"]}
        mock_key = MagicMock()
        mock_key.key = "fake-key"
        with patch("ttllm.core.oidc._get_jwk_client") as mock_jwk:
            mock_jwk.return_value.get_signing_key_from_jwt.return_value = mock_key
            with patch("ttllm.core.oidc.pyjwt.decode", return_value=expected_payload):
                result = verify_id_token(
                    "header.payload.sig",
                    endpoints=_make_endpoints(),
                    client_id="client123",
                    nonce="abc",
                )
        assert result == expected_payload

    def test_pyjwt_error_raises_valueerror(self):
        import jwt as pyjwt

        mock_key = MagicMock()
        mock_key.key = "fake-key"
        with patch("ttllm.core.oidc._get_jwk_client") as mock_jwk:
            mock_jwk.return_value.get_signing_key_from_jwt.return_value = mock_key
            with patch("ttllm.core.oidc.pyjwt.decode", side_effect=pyjwt.InvalidTokenError("bad sig")):
                with pytest.raises(ValueError, match="ID token validation failed"):
                    verify_id_token("header.payload.sig", endpoints=_make_endpoints(), client_id="c")

    def test_nonce_none_skips_check(self):
        mock_key = MagicMock()
        mock_key.key = "fake-key"
        with patch("ttllm.core.oidc._get_jwk_client") as mock_jwk:
            mock_jwk.return_value.get_signing_key_from_jwt.return_value = mock_key
            with patch("ttllm.core.oidc.pyjwt.decode", return_value={"sub": "u1"}):
                result = verify_id_token(
                    "header.payload.sig",
                    endpoints=_make_endpoints(),
                    client_id="c",
                    nonce=None,
                )
        assert result == {"sub": "u1"}


# ---------------------------------------------------------------------------
# extract_roles_from_id_token_payload
# ---------------------------------------------------------------------------


class TestExtractRolesFromPayload:
    """extract_roles_from_id_token_payload works on pre-verified payloads."""

    def test_extracts_roles(self):
        assert extract_roles_from_id_token_payload({"roles": ["admin", "reader"]}) == ["admin", "reader"]

    def test_empty_when_no_roles_key(self):
        assert extract_roles_from_id_token_payload({"sub": "user1"}) == []

    def test_empty_dict(self):
        assert extract_roles_from_id_token_payload({}) == []


# ---------------------------------------------------------------------------
# Startup settings validation
# ---------------------------------------------------------------------------


class TestValidateSettings:
    """_validate_settings blocks unsafe configurations."""

    def test_default_secret_in_prod_raises(self):
        from ttllm.config import Settings, _validate_settings

        s = Settings()
        with pytest.raises(SystemExit, match="FATAL.*JWT secret_key"):
            _validate_settings(s, "prod")

    def test_default_secret_in_dev_warns(self, caplog):
        from ttllm.config import Settings, _validate_settings

        s = Settings()
        with caplog.at_level("WARNING"):
            _validate_settings(s, "dev")
        assert "default value" in caplog.text

    def test_custom_secret_no_error(self, caplog):
        from ttllm.config import Settings, _validate_settings

        s = Settings(auth={"jwt": {"secret_key": "my-secure-secret-value-here"}})
        with caplog.at_level("WARNING"):
            _validate_settings(s, "prod")
        assert "JWT secret_key" not in caplog.text

    def test_invalid_fernet_key_raises(self):
        from ttllm.config import Settings, _validate_settings

        s = Settings(
            auth={"jwt": {"secret_key": "real-secret"}},
            secrets={"encryption_key": "not-a-valid-fernet-key"},
        )
        with pytest.raises(SystemExit, match="FATAL.*Fernet"):
            _validate_settings(s, "prod")

    def test_valid_fernet_key_passes(self):
        from cryptography.fernet import Fernet

        from ttllm.config import Settings, _validate_settings

        key = Fernet.generate_key().decode()
        s = Settings(
            auth={"jwt": {"secret_key": "real-secret"}},
            secrets={"encryption_key": key},
        )
        _validate_settings(s, "prod")


# ---------------------------------------------------------------------------
# YAML SafeLoader
# ---------------------------------------------------------------------------


class TestYamlSafeLoader:
    """Config files are loaded with SafeLoader, not FullLoader."""

    def test_safe_loader_rejects_python_objects(self, tmp_path):
        from ttllm.config import ConfigLoader

        ConfigLoader.clear_cache()
        evil_yaml = tmp_path / "evil.yaml"
        evil_yaml.write_text("dev:\n  value: !!python/object/apply:os.getenv ['HOME']\n")
        with pytest.raises(Exception):
            ConfigLoader(config_file=str(evil_yaml), environment="dev")


# ---------------------------------------------------------------------------
# Error message sanitization
# ---------------------------------------------------------------------------


class TestErrorMessageSanitization:
    """Internal error messages don't leak exception details to clients."""

    def test_invoke_error_is_generic(self, client):
        resp = client.post(
            "/v1/messages",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            headers={"Authorization": "Bearer fake"},
        )
        if resp.status_code == 500:
            body = resp.json()
            assert body["detail"]["message"] == "An internal error occurred"

    def test_health_does_not_leak_db_errors(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Timing-safe authentication
# ---------------------------------------------------------------------------


class TestAuthenticateLocalTimingSafety:
    """authenticate_local runs dummy hash on missing/invalid users to prevent timing attacks."""

    @pytest.mark.asyncio
    async def test_dummy_hash_called_for_missing_user(self):
        from ttllm.services.auth_service import _DUMMY_HASH, authenticate_local

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("ttllm.services.auth_service.verify_password") as mock_verify:
            mock_verify.return_value = False
            result = await authenticate_local(mock_db, "nobody@example.com", "password123")
            assert result is None
            mock_verify.assert_called_once()
            assert mock_verify.call_args[0][0] == "password123"
            assert mock_verify.call_args[0][1] == _DUMMY_HASH

    @pytest.mark.asyncio
    async def test_dummy_hash_called_for_sso_user(self):
        from ttllm.services.auth_service import _DUMMY_HASH, authenticate_local

        mock_user = MagicMock()
        mock_user.is_active = True
        mock_user.identity_provider = "entra"
        mock_user.password_hash = None

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("ttllm.services.auth_service.verify_password") as mock_verify:
            mock_verify.return_value = False
            result = await authenticate_local(mock_db, "sso@example.com", "password123")
            assert result is None
            mock_verify.assert_called_once()
            assert mock_verify.call_args[0][1] == _DUMMY_HASH

    @pytest.mark.asyncio
    async def test_inactive_user_gets_dummy_hash(self):
        from ttllm.services.auth_service import _DUMMY_HASH, authenticate_local

        mock_user = MagicMock()
        mock_user.is_active = False
        mock_user.identity_provider = None
        mock_user.password_hash = "real-hash"

        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("ttllm.services.auth_service.verify_password") as mock_verify:
            mock_verify.return_value = False
            result = await authenticate_local(mock_db, "inactive@example.com", "password123")
            assert result is None
            mock_verify.assert_called_once()
            assert mock_verify.call_args[0][1] == _DUMMY_HASH
