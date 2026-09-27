"""LLM model and assignment CRUD operations."""

from __future__ import annotations

import re
import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.models.auth import UserGroup
from ttllm.models.llm_model import GroupModelAssignment, LLMModel, ModelAssignment


async def create_model(
    db: AsyncSession,
    name: str,
    provider: str,
    provider_model_id: str,
    display_name: str | None = None,
    config_json: dict | None = None,
    input_cost_per_1k: Decimal = Decimal("0"),
    output_cost_per_1k: Decimal = Decimal("0"),
    cache_read_cost_per_1k: Decimal = Decimal("0"),
    cache_write_cost_per_1k: Decimal = Decimal("0"),
    match_pattern: str | None = None,
) -> LLMModel:
    model = LLMModel(
        name=name,
        display_name=display_name,
        provider=provider,
        provider_model_id=provider_model_id,
        config_json=config_json or {},
        input_cost_per_1k=input_cost_per_1k,
        output_cost_per_1k=output_cost_per_1k,
        cache_read_cost_per_1k=cache_read_cost_per_1k,
        cache_write_cost_per_1k=cache_write_cost_per_1k,
        match_pattern=match_pattern,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    return model


async def get_model(db: AsyncSession, model_id: uuid.UUID) -> LLMModel | None:
    return await db.get(LLMModel, model_id)


async def get_model_by_name(db: AsyncSession, name: str) -> LLMModel | None:
    result = await db.execute(
        select(LLMModel).where(LLMModel.name == name, LLMModel.is_active == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


async def list_models(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
    include_inactive: bool = False,
) -> tuple[list[LLMModel], int]:
    query = select(LLMModel)
    if not include_inactive:
        query = query.where(LLMModel.is_active == True)  # noqa: E712

    count_result = await db.execute(select(LLMModel.id).where(LLMModel.is_active == True))  # noqa: E712
    total = len(count_result.all())

    query = query.offset(offset).limit(limit).order_by(LLMModel.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_model(
    db: AsyncSession,
    model_id: uuid.UUID,
    **kwargs,
) -> LLMModel | None:
    model = await db.get(LLMModel, model_id)
    if not model:
        return None
    if kwargs.pop("merge_config", False) and "config_json" in kwargs:
        merged = {**(model.config_json or {}), **kwargs.pop("config_json")}
        setattr(model, "config_json", merged)
    # "name" is deliberately not mutable here -- it's the model's immutable identity/
    # routing key (audit_logs.model_name is grouped/searched on it). "display_name" is
    # the mutable, human-friendly label. ModelUpdate doesn't even expose "name", but the
    # API layer passes **kwargs straight from a dict, so this is the actual enforcement.
    _MUTABLE_FIELDS = {"display_name", "provider", "provider_model_id", "config_json", "input_cost_per_1k", "output_cost_per_1k", "cache_read_cost_per_1k", "cache_write_cost_per_1k", "is_active", "match_pattern"}
    for key, value in kwargs.items():
        if key in _MUTABLE_FIELDS and value is not None:
            setattr(model, key, value)
    await db.commit()
    await db.refresh(model)
    return model


async def delete_model(db: AsyncSession, model_id: uuid.UUID) -> bool:
    """Permanently remove a model.

    This is a real delete, not a deactivation — ``is_active`` (settable via
    ``update_model``) is the on/off switch for pausing a model without losing it.
    Safe unconditionally: model_assignments/group_model_assignments cascade at the DB
    level, and audit_logs has no FK to llm_models (it keeps a self-contained
    model_name/model_snapshot instead), so past usage history is unaffected.
    """
    model = await db.get(LLMModel, model_id)
    if not model:
        return False
    await db.delete(model)
    await db.commit()
    return True


def build_model_snapshot(model: LLMModel) -> dict:
    """Point-in-time dump of a model, for audit_logs.model_snapshot.

    Captured at request time so audit history stays readable/accurate even after the
    model is later renamed or deleted. Not redacted: config_json only ever holds
    ``secret://`` references (resolution happens transiently, never persisted), and
    what an admin puts in it is their call, not something we should second-guess here.
    """
    return {
        "id": str(model.id),
        "name": model.name,
        "display_name": model.display_name,
        "provider": model.provider,
        "provider_model_id": model.provider_model_id,
        "config_json": model.config_json,
        "input_cost_per_1k": str(model.input_cost_per_1k),
        "output_cost_per_1k": str(model.output_cost_per_1k),
        "cache_read_cost_per_1k": str(model.cache_read_cost_per_1k),
        "cache_write_cost_per_1k": str(model.cache_write_cost_per_1k),
        "match_pattern": model.match_pattern,
        "is_active": model.is_active,
    }


# --- Assignments ---


async def assign_model_to_user(
    db: AsyncSession,
    model_id: uuid.UUID,
    user_id: uuid.UUID,
) -> ModelAssignment:
    assignment = ModelAssignment(user_id=user_id, model_id=model_id)
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)
    return assignment


async def unassign_model_from_user(
    db: AsyncSession,
    model_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    result = await db.execute(
        select(ModelAssignment).where(
            ModelAssignment.model_id == model_id,
            ModelAssignment.user_id == user_id,
        )
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        return False
    await db.delete(assignment)
    await db.commit()
    return True


async def assign_model_to_group(
    db: AsyncSession,
    model_id: uuid.UUID,
    group_id: uuid.UUID,
) -> GroupModelAssignment:
    assignment = GroupModelAssignment(group_id=group_id, model_id=model_id)
    db.add(assignment)
    await db.commit()
    await db.refresh(assignment)
    return assignment


async def unassign_model_from_group(
    db: AsyncSession,
    model_id: uuid.UUID,
    group_id: uuid.UUID,
) -> bool:
    result = await db.execute(
        select(GroupModelAssignment).where(
            GroupModelAssignment.model_id == model_id,
            GroupModelAssignment.group_id == group_id,
        )
    )
    assignment = result.scalar_one_or_none()
    if not assignment:
        return False
    await db.delete(assignment)
    await db.commit()
    return True


async def get_model_for_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    model_name: str,
) -> LLMModel | None:
    """Get a model by name if the user has access (direct or via group).

    Resolution order:
    1. Exact name match
    2. Regex pattern match (fallback via match_pattern column)
    """
    direct = (
        select(LLMModel.id)
        .join(ModelAssignment, ModelAssignment.model_id == LLMModel.id)
        .where(ModelAssignment.user_id == user_id)
    )
    via_group = (
        select(LLMModel.id)
        .join(GroupModelAssignment, GroupModelAssignment.model_id == LLMModel.id)
        .join(UserGroup, UserGroup.group_id == GroupModelAssignment.group_id)
        .where(UserGroup.user_id == user_id)
    )
    result = await db.execute(
        select(LLMModel).where(
            LLMModel.id.in_(direct.union(via_group)),
            LLMModel.name == model_name,
            LLMModel.is_active == True,  # noqa: E712
        )
    )
    exact = result.scalar_one_or_none()
    if exact:
        return exact

    all_models = await list_user_models(db, user_id)
    for model in all_models:
        if not model.match_pattern:
            continue
        try:
            if re.fullmatch(model.match_pattern, model_name):
                return model
        except re.error:
            continue
    return None


async def list_user_models(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[LLMModel]:
    """List all models assigned to a user (direct or via group)."""
    direct = (
        select(LLMModel.id)
        .join(ModelAssignment, ModelAssignment.model_id == LLMModel.id)
        .where(ModelAssignment.user_id == user_id)
    )
    via_group = (
        select(LLMModel.id)
        .join(GroupModelAssignment, GroupModelAssignment.model_id == LLMModel.id)
        .join(UserGroup, UserGroup.group_id == GroupModelAssignment.group_id)
        .where(UserGroup.user_id == user_id)
    )
    result = await db.execute(
        select(LLMModel).where(
            LLMModel.id.in_(direct.union(via_group)),
            LLMModel.is_active == True,  # noqa: E712
        )
    )
    return list(result.scalars().all())
