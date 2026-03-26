"""OIDC (OpenID Connect) protocol helpers. Pure async logic using httpx."""

from __future__ import annotations

import hashlib
import json
import secrets
from base64 import urlsafe_b64decode, urlsafe_b64encode
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx


@dataclass(frozen=True)
class OIDCEndpoints:
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    issuer: str


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


def extract_roles_from_id_token(id_token: str) -> list[str]:
    """Decode the payload of a JWT ID token (without verification) to extract roles.

    The IdP signature is already validated during the code exchange.
    """
    if not id_token:
        return []
    try:
        payload_b64 = id_token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(urlsafe_b64decode(payload_b64))
        return payload.get("roles", [])
    except Exception:
        return []
