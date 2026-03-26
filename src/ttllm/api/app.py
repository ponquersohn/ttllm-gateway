"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ttllm.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="TTLLM Gateway",
        description="LLM Gateway with Anthropic-compatible API",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.engine.cors_origins,
        allow_credentials=True,
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
    from ttllm.api.messages import router as messages_router

    app.include_router(auth_router)
    app.include_router(messages_router)
    app.include_router(admin_router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
