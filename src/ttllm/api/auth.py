"""Authentication endpoints: login, SSO, token refresh, logout."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from ttllm.api.deps import AuthContext, DB, get_authenticated
from ttllm.config import settings
from ttllm.core import oidc
from ttllm.schemas.auth import LoginRequest, LoginTokenResponse, RefreshRequest
from ttllm.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory store for OIDC state -> (code_verifier, idp_slug, redirect_uri)
# In production, use Redis or a DB table with TTL.
_oidc_state_store: dict[str, dict] = {}


@router.get("/identity-providers")
async def list_identity_providers():
    """Return configured identity providers (public, no auth required)."""
    return [
        {"slug": slug, "name": idp.name, "type": idp.type}
        for slug, idp in settings.auth.identity_providers.items()
    ]


@router.post("/token", response_model=LoginTokenResponse)
async def login(body: LoginRequest, db: DB):
    """Authenticate with email + password, receive management JWT + refresh token."""
    user = await auth_service.authenticate_local(db, body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid credentials"},
        )
    return await auth_service.create_management_tokens(db, user)


@router.post("/token/refresh", response_model=LoginTokenResponse)
async def refresh(body: RefreshRequest, db: DB):
    """Refresh a management JWT using a refresh token."""
    result = await auth_service.refresh_management_token(db, body.refresh_token)
    if not result:
        raise HTTPException(
            status_code=401,
            detail={"type": "authentication_error", "message": "Invalid or expired refresh token"},
        )
    return result


@router.post("/logout", status_code=204)
async def logout(
    body: RefreshRequest,
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Revoke a refresh token."""
    from ttllm.core.jwt import hash_refresh_token
    from ttllm.models.auth import RefreshToken
    from sqlalchemy import select
    from datetime import UTC, datetime

    token_hash = hash_refresh_token(body.refresh_token)
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.user_id == ctx.user.id,
            RefreshToken.revoked_at.is_(None),
        )
    )
    rt = result.scalar_one_or_none()
    if rt:
        rt.revoked_at = datetime.now(UTC)
        await db.commit()


# --- SSO (OIDC) ---


@router.get("/sso/{idp_slug}/authorize")
async def sso_authorize(idp_slug: str, final_redirect: str | None = None):
    """Start the OIDC authorization code flow. Redirects to the IdP.

    If final_redirect is provided (e.g. by the CLI), the callback will redirect
    there with tokens as query params instead of returning JSON.
    """
    idp_config = settings.auth.identity_providers.get(idp_slug)
    if not idp_config:
        raise HTTPException(404, detail={"type": "not_found", "message": f"Identity provider '{idp_slug}' not found"})

    endpoints = await oidc.discover(idp_config.get_discovery_url())
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    code_verifier, code_challenge = oidc.generate_pkce()

    # The redirect_uri sent to the IdP must match what's registered
    base = settings.engine.base_url.rstrip("/")
    callback_uri = f"{base}/auth/sso/{idp_slug}/callback"

    _oidc_state_store[state] = {
        "code_verifier": code_verifier,
        "idp_slug": idp_slug,
        "redirect_uri": callback_uri,
        "nonce": nonce,
        "final_redirect": final_redirect,
    }

    auth_url = oidc.build_authorization_url(
        endpoints=endpoints,
        client_id=idp_config.client_id,
        redirect_uri=callback_uri,
        state=state,
        nonce=nonce,
        scopes=idp_config.scopes,
        code_challenge=code_challenge,
    )
    return RedirectResponse(url=auth_url)


@router.get("/sso/{idp_slug}/callback")
async def sso_callback(
    idp_slug: str,
    code: str,
    state: str,
    db: DB,
):
    """OIDC callback: exchange code for tokens, provision user, return JWT.

    If the authorize step included a final_redirect (CLI flow), redirects to that
    URL with access_token and refresh_token as query params. Otherwise returns JSON.
    """
    state_data = _oidc_state_store.pop(state, None)
    if not state_data or state_data["idp_slug"] != idp_slug:
        raise HTTPException(400, detail={"type": "invalid_request", "message": "Invalid or expired state"})

    idp_config = settings.auth.identity_providers.get(idp_slug)
    if not idp_config:
        raise HTTPException(404, detail={"type": "not_found", "message": f"Identity provider '{idp_slug}' not found"})

    endpoints = await oidc.discover(idp_config.get_discovery_url())

    # Exchange code for IdP tokens
    token_data = await oidc.exchange_code(
        endpoints=endpoints,
        client_id=idp_config.client_id,
        client_secret=idp_config.client_secret,
        code=code,
        redirect_uri=state_data["redirect_uri"],
        code_verifier=state_data["code_verifier"],
    )

    # Fetch user info
    idp_access_token = token_data.get("access_token", "")
    user_info = await oidc.fetch_userinfo(endpoints, idp_access_token)

    # Extract roles from ID token and map to ttllm group names
    idp_roles = oidc.extract_roles_from_id_token(token_data.get("id_token", ""))
    target_groups = set(idp_config.default_groups)
    for idp_role in idp_roles:
        target_groups.update(idp_config.group_mapping.get(idp_role, []))

    # Provision or look up user, sync group memberships
    user = await auth_service.provision_sso_user(
        db,
        idp_slug=idp_slug,
        user_info=user_info,
        target_groups=target_groups,
    )

    result = await auth_service.create_management_tokens(db, user)

    # CLI flow: redirect back to the CLI's ephemeral server with tokens
    final_redirect = state_data.get("final_redirect")
    if final_redirect:
        from urllib.parse import urlencode
        params = urlencode({
            "access_token": result.access_token,
            "refresh_token": result.refresh_token or "",
        })
        return RedirectResponse(url=f"{final_redirect}?{params}")

    return result
