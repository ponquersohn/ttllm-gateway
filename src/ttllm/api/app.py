"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.staticfiles import StaticFiles

from ttllm import __version__
from ttllm.config import settings


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-store",
}


class _SecurityHeadersMiddleware:
    """Pure ASGI middleware that injects security headers on every response.

    Implemented at the ASGI layer (rather than via BaseHTTPMiddleware) so it does
    not wrap responses through an extra task/cancel scope, which buffers
    StreamingResponse output and breaks incremental SSE streaming.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                for key, value in _SECURITY_HEADERS.items():
                    headers[key] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _authenticated_docs_html(variant: str, title: str) -> HTMLResponse:
    """Return Swagger UI or ReDoc HTML that reads the JWT from sessionStorage."""
    if variant == "swagger":
        body = """
    <div id="swagger-ui"></div>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
    SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
      layout: "StandaloneLayout",
      requestInterceptor: function(req) {
        req.headers["Authorization"] = "Bearer " + sessionStorage.getItem("access_token");
        return req;
      }
    });
    </script>"""
    else:
        body = """
    <script>
    // Intercept fetch BEFORE ReDoc loads so the spec request includes auth
    (function() {
      var _fetch = window.fetch;
      window.fetch = function(url, opts) {
        if (typeof url === "string" && url.includes("/openapi.json")) {
          opts = opts || {};
          opts.headers = opts.headers || {};
          opts.headers["Authorization"] = "Bearer " + sessionStorage.getItem("access_token");
        }
        return _fetch.call(this, url, opts);
      };
    })();
    </script>
    <redoc spec-url="/openapi.json"></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>"""

    return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
  <title>{title} - API Docs</title>
  <meta charset="utf-8">
  <script>
    if (!sessionStorage.getItem("access_token")) {{
      window.location.href = "/ui?return_to=" + encodeURIComponent(window.location.pathname);
    }}
  </script>
</head><body>
{body}
</body></html>""")


def create_app() -> FastAPI:
    app = FastAPI(
        title="TTLLM Gateway",
        description="LLM Gateway with Anthropic-compatible API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.add_middleware(_SecurityHeadersMiddleware)

    # CORS — disable credentials when origins include wildcard
    origins = settings.engine.cors_origins
    allow_creds = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers matching Anthropic error format
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            error_type = detail.get("type", "api_error")
            message = detail.get("message", str(detail))
        else:
            error_type = "api_error"
            message = str(detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "type": "error",
                "error": {"type": error_type, "message": message},
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "Internal server error",
                },
            },
        )

    # Initialize permission registry
    from ttllm.core.permissions import Permissions
    from ttllm.services import auth_service

    auth_service.set_permission_registry(Permissions.get_registry())

    # Include routers
    from ttllm.api.admin import router as admin_router
    from ttllm.api.auth import router as auth_router
    from ttllm.api.me import router as me_router
    from ttllm.api.messages import router as messages_router

    app.include_router(auth_router)
    app.include_router(messages_router)
    app.include_router(me_router)
    app.include_router(admin_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    # --- Authenticated API docs ---
    from ttllm.api.deps import get_authenticated

    @app.get("/openapi.json", include_in_schema=False)
    async def openapi_schema(_: None = Depends(get_authenticated)):
        return app.openapi()

    @app.get("/docs", include_in_schema=False)
    async def docs_page():
        return _authenticated_docs_html("swagger", app.title)

    @app.get("/redoc", include_in_schema=False)
    async def redoc_page():
        return _authenticated_docs_html("redoc", app.title)

    # Mount self-service UI (static files)
    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    if ui_dir.is_dir():

        @app.get("/ui", include_in_schema=False)
        async def ui_redirect(request: Request):
            from starlette.responses import RedirectResponse

            query = str(request.url.query)
            target = "/ui/" + ("?" + query if query else "")
            return RedirectResponse(url=target)

        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app
