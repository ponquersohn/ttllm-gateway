"""GET /anthropic/v1/models - Anthropic-compatible model discovery."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ttllm.api.deps import DB
from ttllm.api.messages import AnthropicUser
from ttllm.schemas.anthropic import AnthropicModelListResponse, AnthropicModelObject
from ttllm.services import model_service

router = APIRouter(tags=["models"])


def _to_anthropic_model(model) -> AnthropicModelObject:
    return AnthropicModelObject(
        id=model.name,
        display_name=model.display_name or model.name,
        created_at=model.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


@router.get("/anthropic/v1/models", response_model=AnthropicModelListResponse)
async def list_models(
    db: DB,
    ctx: AnthropicUser,
    limit: int = Query(default=20, ge=1, le=1000),
    after_id: str | None = Query(default=None),
    before_id: str | None = Query(default=None),
):
    """List models available to the authenticated user."""
    all_models = await model_service.list_user_models(db, ctx.user.id)
    all_models.sort(key=lambda m: m.name)

    if after_id:
        idx = next((i for i, m in enumerate(all_models) if m.name == after_id), None)
        if idx is not None:
            all_models = all_models[idx + 1 :]
        else:
            all_models = []
    elif before_id:
        idx = next((i for i, m in enumerate(all_models) if m.name == before_id), None)
        if idx is not None:
            all_models = all_models[:idx]
        else:
            all_models = []

    has_more = len(all_models) > limit
    page = all_models[:limit]
    data = [_to_anthropic_model(m) for m in page]

    return AnthropicModelListResponse(
        data=data,
        has_more=has_more,
        first_id=data[0].id if data else None,
        last_id=data[-1].id if data else None,
    )


@router.get("/anthropic/v1/models/{model_id:path}", response_model=AnthropicModelObject)
async def get_model(
    model_id: str,
    db: DB,
    ctx: AnthropicUser,
):
    """Get a single model by ID."""
    model = await model_service.get_model_for_user(db, ctx.user.id, model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail={"type": "not_found_error", "message": f"Model '{model_id}' not found"},
        )
    return _to_anthropic_model(model)
