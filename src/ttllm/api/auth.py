"""Authentication endpoints: login, SSO, token refresh, logout."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from ttllm.api.deps import AuthContext, DB, get_authenticated
from ttllm.config import settings
from ttllm.core import oidc
from ttllm.core.device_flow import PollOutcome
from ttllm.schemas.auth import (
    DeviceApproveRequest,
    DeviceCodeRequest,
    DeviceCodeResponse,
    DeviceLookupResponse,
    DeviceTokenRequest,
    LoginRequest,
    LoginTokenResponse,
    RefreshRequest,
)
from ttllm.services import auth_service, device_auth_service, oidc_state_service

router = APIRouter(prefix="/auth", tags=["auth"])

_LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _validate_redirect(url: str) -> bool:
    """Allow redirects to localhost (always) and configured origins."""
    try:
        parsed = urlparse(url)
        if parsed.hostname in _LOCALHOST_HOSTS:
            return True
        target_origin = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            target_origin += f":{parsed.port}"
        return target_origin in settings.auth.allowed_redirect_origins
    except Exception:
        return False


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
    try:
        user = await auth_service.authenticate_local(db, body.email, body.password)
    except auth_service.InactiveUserError:
        raise HTTPException(
            status_code=403,
            detail={"type": "permission_error", "message": auth_service.INACTIVE_USER_MESSAGE},
        )
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


# --- Device code flow (RFC 8628) ---

_DEVICE_POLL_MESSAGES = {
    PollOutcome.AUTHORIZATION_PENDING: "The authorization request is still pending",
    PollOutcome.SLOW_DOWN: "Polling too fast; increase the interval by 5 seconds",
    PollOutcome.ACCESS_DENIED: "The authorization request was denied",
    PollOutcome.EXPIRED_TOKEN: "The device code has expired",
}


def _device_not_found() -> HTTPException:
    return HTTPException(
        404, detail={"type": "not_found", "message": "Unknown or expired device code"}
    )


@router.post("/device/code", response_model=DeviceCodeResponse)
async def device_code(body: DeviceCodeRequest, request: Request, db: DB):
    """Start a device code login. The CLI shows `user_code` and polls /auth/device/token."""
    row, raw_device_code, user_code = await device_auth_service.create(
        db,
        client_name=body.client_name,
        requester_ip=request.client.host if request.client else None,
        requester_user_agent=request.headers.get("user-agent"),
    )
    await db.commit()

    base = settings.engine.base_url.rstrip("/")
    verification_uri = f"{base}/ui/?device=1"
    return DeviceCodeResponse(
        device_code=raw_device_code,
        user_code=user_code,
        verification_uri=verification_uri,
        verification_uri_complete=f"{verification_uri}&user_code={user_code}",
        expires_in=device_auth_service.expires_in(row),
        interval=row.interval_seconds,
    )


@router.post("/device/token", response_model=LoginTokenResponse)
async def device_token(body: DeviceTokenRequest, db: DB):
    """Poll for the result of a device code login."""
    outcome, user = await device_auth_service.poll(db, body.device_code)
    await db.commit()

    if outcome is None:
        # Unknown, already used, or swept: indistinguishable from expiry to the client.
        outcome = PollOutcome.EXPIRED_TOKEN
    if outcome != PollOutcome.APPROVED:
        raise HTTPException(
            400, detail={"type": outcome.value, "message": _DEVICE_POLL_MESSAGES[outcome]}
        )
    assert user is not None
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail={"type": "permission_error", "message": auth_service.INACTIVE_USER_MESSAGE},
        )
    return await auth_service.create_management_tokens(db, user)


async def _require_session(ctx: AuthContext, db) -> None:
    """Device approval needs an interactive login session, not a gateway API token."""
    if await auth_service.token_exists(db, ctx.jti):
        raise HTTPException(
            403,
            detail={
                "type": "permission_error",
                "message": "Device logins must be approved from a login session, not an API token",
            },
        )


@router.get("/device/lookup", response_model=DeviceLookupResponse)
async def device_lookup(
    user_code: str,
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Describe a pending device login so the user can confirm it before approving."""
    await _require_session(ctx, db)
    row = await device_auth_service.get_pending_by_user_code(db, user_code)
    if row is None:
        raise _device_not_found()
    response = DeviceLookupResponse(
        client_name=row.client_name,
        requester_ip=row.requester_ip,
        requester_user_agent=row.requester_user_agent,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )
    await db.rollback()  # release the row lock taken by the lookup
    return response


@router.post("/device/approve", status_code=204)
async def device_approve(
    body: DeviceApproveRequest,
    db: DB,
    ctx: AuthContext = Depends(get_authenticated),
):
    """Approve (or deny) a pending device login as the authenticated user."""
    await _require_session(ctx, db)
    if not await device_auth_service.resolve(db, body.user_code, ctx.user, body.approve):
        raise _device_not_found()
    await db.commit()


# --- SSO (OIDC) ---


@router.get("/sso/{idp_slug}/authorize")
async def sso_authorize(idp_slug: str, db: DB, final_redirect: str | None = None):
    """Start the OIDC authorization code flow. Redirects to the IdP.

    If final_redirect is provided (e.g. by the CLI), the callback will redirect
    there with tokens as query params instead of returning JSON.
    """
    idp_config = settings.auth.identity_providers.get(idp_slug)
    if not idp_config:
        raise HTTPException(404, detail={"type": "not_found", "message": f"Identity provider '{idp_slug}' not found"})

    if final_redirect and not _validate_redirect(final_redirect):
        logger = logging.getLogger(__name__)
        parsed = urlparse(final_redirect)
        target_origin = f"{parsed.scheme}://{parsed.hostname}"
        if parsed.port:
            target_origin += f":{parsed.port}"
        logger.warning(
            "Rejected final_redirect=%s (origin=%s). allowed_redirect_origins=%r",
            final_redirect, target_origin, settings.auth.allowed_redirect_origins,
        )
        raise HTTPException(400, detail={"type": "invalid_request", "message": "final_redirect must point to localhost"})

    endpoints = await oidc.discover(idp_config.get_discovery_url())
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    code_verifier, code_challenge = oidc.generate_pkce()

    base = settings.engine.base_url.rstrip("/")
    callback_uri = f"{base}/auth/sso/{idp_slug}/callback"

    await oidc_state_service.store_state(db, state, {
        "code_verifier": code_verifier,
        "idp_slug": idp_slug,
        "redirect_uri": callback_uri,
        "nonce": nonce,
        "final_redirect": final_redirect,
    })
    await db.commit()

    auth_url = oidc.build_authorization_url(
        endpoints=endpoints,
        client_id=idp_config.client_id,
        redirect_uri=callback_uri,
        state=state,
        nonce=nonce,
        scopes=list(idp_config.scopes) + (["offline_access"] if "offline_access" not in idp_config.scopes else []),
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
    state_data = await oidc_state_service.pop_state(db, state)
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

    # Verify ID token signature + nonce, then extract roles
    id_token_raw = token_data.get("id_token", "")
    target_groups = set(idp_config.default_groups)
    if id_token_raw:
        try:
            id_payload = oidc.verify_id_token(
                id_token_raw,
                endpoints=endpoints,
                client_id=idp_config.client_id,
                nonce=state_data.get("nonce"),
            )
        except ValueError as exc:
            raise HTTPException(400, detail={"type": "invalid_request", "message": f"ID token verification failed: {exc}"})
        idp_roles = oidc.extract_roles_from_id_token_payload(id_payload)
        for idp_role in idp_roles:
            target_groups.update(idp_config.group_mapping.get(idp_role, []))

    if not target_groups:
        raise HTTPException(
            status_code=403,
            detail={
                "type": "access_denied",
                "message": "You do not have any roles assigned in your organization. Contact your administrator to request access.",
            },
        )

    # Compute the full set of groups this IdP can manage
    sso_managed_groups = set(idp_config.default_groups)
    for mapped in idp_config.group_mapping.values():
        sso_managed_groups.update(mapped)

    # Provision or look up user, sync group memberships
    user = await auth_service.provision_sso_user(
        db,
        idp_slug=idp_slug,
        user_info=user_info,
        target_groups=target_groups,
        sso_managed_groups=sso_managed_groups,
        idp_refresh_token=token_data.get("refresh_token"),
    )

    final_redirect = state_data.get("final_redirect")

    # Deactivated accounts can still authenticate at the IdP, but must not be
    # issued gateway tokens. Tell the caller why instead of handing out tokens
    # that would 403 on every request.
    if not user.is_active:
        if final_redirect:
            from urllib.parse import urlencode
            params = urlencode({
                "error": "account_inactive",
                "error_description": auth_service.INACTIVE_USER_MESSAGE,
            })
            return RedirectResponse(url=f"{final_redirect}?{params}")
        raise HTTPException(
            status_code=403,
            detail={"type": "permission_error", "message": auth_service.INACTIVE_USER_MESSAGE},
        )

    result = await auth_service.create_management_tokens(db, user)

    # CLI flow: redirect back to the CLI's ephemeral server with tokens
    if final_redirect:
        from urllib.parse import urlencode
        params = urlencode({
            "access_token": result.access_token,
            "refresh_token": result.refresh_token or "",
        })
        return RedirectResponse(url=f"{final_redirect}?{params}")

    return result
