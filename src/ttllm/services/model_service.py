"""LLM model and assignment CRUD operations."""

from __future__ import annotations

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
    config_json: dict | None = None,
    input_cost_per_1k: Decimal = Decimal("0"),
    output_cost_per_1k: Decimal = Decimal("0"),
) -> LLMModel:
    model = LLMModel(
        name=name,
        provider=provider,
        provider_model_id=provider_model_id,
        config_json=config_json or {},
        input_cost_per_1k=input_cost_per_1k,
        output_cost_per_1k=output_cost_per_1k,
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
    for key, value in kwargs.items():
        if hasattr(model, key) and value is not None:
            setattr(model, key, value)
    await db.commit()
    await db.refresh(model)
    return model


async def delete_model(db: AsyncSession, model_id: uuid.UUID) -> bool:
    model = await db.get(LLMModel, model_id)
    if not model:
        return False
    model.is_active = False
    await db.commit()
    return True


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
    """Get a model by name if the user has access (direct or via group)."""
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
    return result.scalar_one_or_none()


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
