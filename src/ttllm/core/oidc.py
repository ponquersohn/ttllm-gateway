"""OIDC (OpenID Connect) protocol helpers. Pure async logic using httpx."""

from __future__ import annotations

import hashlib
import logging
import secrets
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import jwt as pyjwt
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

_jwk_clients: dict[str, PyJWKClient] = {}


def _get_jwk_client(jwks_uri: str) -> PyJWKClient:
    """Return a cached PyJWKClient for the given JWKS URI."""
    if jwks_uri not in _jwk_clients:
        _jwk_clients[jwks_uri] = PyJWKClient(jwks_uri, cache_jwk_set=True, lifespan=300)
    return _jwk_clients[jwks_uri]


@dataclass(frozen=True)
class OIDCEndpoints:
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    issuer: str
    jwks_uri: str


async def discover(discovery_url: str) -> OIDCEndpoints:
    """Fetch the OIDC provider's .well-known/openid-configuration."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(discovery_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    return OIDCEndpoints(
        authorization_endpoint=data["authorization_endpoint"],
        token_endpoint=data["token_endpoint"],
        userinfo_endpoint=data["userinfo_endpoint"],
        issuer=data["issuer"],
        jwks_uri=data["jwks_uri"],
    )


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge pair."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def build_authorization_url(
    endpoints: OIDCEndpoints,
    client_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    scopes: list[str],
    code_challenge: str,
) -> str:
    """Construct the full OIDC authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"{endpoints.authorization_endpoint}?{urlencode(params)}"


async def exchange_code(
    endpoints: OIDCEndpoints,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> dict:
    """Exchange an authorization code for tokens at the IdP token endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            endpoints.token_endpoint,
            data={
                "grant_type": "authorization_code",
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


async def fetch_userinfo(
    endpoints: OIDCEndpoints,
    access_token: str,
) -> dict:
    """Fetch user claims from the IdP userinfo endpoint."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            endpoints.userinfo_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()


def verify_id_token(
    id_token: str,
    endpoints: OIDCEndpoints,
    client_id: str,
    nonce: str | None = None,
) -> dict:
    """Verify the ID token signature against the IdP JWKS and return the payload.

    Validates: signature, expiry, issuer, audience, and (optionally) nonce.
    Raises ValueError on any verification failure.
    """
    if not id_token:
        raise ValueError("Empty ID token")
    try:
        client = _get_jwk_client(endpoints.jwks_uri)
        signing_key = client.get_signing_key_from_jwt(id_token)

        payload = pyjwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
            audience=client_id,
            issuer=endpoints.issuer,
        )
    except pyjwt.PyJWTError as exc:
        raise ValueError(f"ID token validation failed: {exc}") from exc

    if nonce is not None and payload.get("nonce") != nonce:
        raise ValueError("ID token nonce mismatch")

    return payload


def extract_roles_from_id_token_payload(payload: dict) -> list[str]:
    """Extract roles from an already-verified ID token payload."""
    return payload.get("roles", [])
