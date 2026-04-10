"""HTTP client with transparent token refresh for the TTLLM CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

SESSION_DIR = Path.home() / ".config" / "ttllm"
SESSION_FILE = SESSION_DIR / "session.json"


class TTLLMClient(httpx.Client):
    """httpx.Client subclass that automatically refreshes expired tokens."""

    _refreshed: bool = False

    # --- Session management ---

    @staticmethod
    def load_session() -> dict | None:
        if SESSION_FILE.exists():
            return json.loads(SESSION_FILE.read_text())
        return None

    @staticmethod
    def save_session(data: dict) -> None:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(data, indent=2))

    @staticmethod
    def clear_session() -> None:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()

    @staticmethod
    def base_url_from_session() -> str:
        session = TTLLMClient.load_session()
        if session and session.get("base_url"):
            return session["base_url"]
        return os.environ.get("TTLLM_URL", "http://localhost:4000")

    @classmethod
    def from_session(cls) -> TTLLMClient:
        """Create a client from the stored session. Returns None-ish — callers
        should check for access_token before calling this."""
        base_url = cls.base_url_from_session()
        session = cls.load_session()
        token = session["access_token"] if session else None
        return cls(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=30,
        )

    # --- Auth ---

    @classmethod
    def login(cls, base_url: str, email: str, password: str) -> httpx.Response:
        """Authenticate with email/password and persist the session."""
        resp = httpx.post(
            f"{base_url}/auth/token",
            json={"email": email, "password": password},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            cls.save_session({
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token"),
                "base_url": base_url,
            })
        return resp

    @classmethod
    def login_with_tokens(cls, base_url: str, access_token: str, refresh_token: str | None = None) -> None:
        """Persist a session from externally-obtained tokens (e.g. SSO)."""
        cls.save_session({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "base_url": base_url,
        })

    def logout(self) -> None:
        """Revoke the refresh token (best-effort) and clear the local session."""
        session = self.load_session()
        if session and session.get("refresh_token"):
            try:
                self.post("/auth/logout", json={"refresh_token": session["refresh_token"]})
            except Exception:
                pass
        self.clear_session()

    # --- Request with auto-refresh ---

    def request(self, method: str, url, **kwargs) -> httpx.Response:
        resp = super().request(method, url, **kwargs)
        if resp.status_code != 401 or self._refreshed:
            return resp

        # Attempt token refresh
        session = self.load_session()
        if not session or not session.get("refresh_token"):
            return resp
        refresh_resp = httpx.post(
            f"{self.base_url}auth/token/refresh",
            json={"refresh_token": session["refresh_token"]},
            timeout=10,
        )
        if refresh_resp.status_code != 200:
            return resp

        data = refresh_resp.json()
        session["access_token"] = data["access_token"]
        session["refresh_token"] = data.get("refresh_token", session["refresh_token"])
        self.save_session(session)

        self.headers["authorization"] = f"Bearer {session['access_token']}"
        self._refreshed = True
        return super().request(method, url, **kwargs)
