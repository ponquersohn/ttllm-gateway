"""Group, permission assignment, and membership operations."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ttllm.models.auth import Group, GroupPermission, UserGroup
from ttllm.models.user import User


async def create_group(
    db: AsyncSession,
    name: str,
    description: str | None = None,
) -> Group:
    group = Group(name=name, description=description)
    db.add(group)
    await db.commit()
    await db.refresh(group, attribute_names=["permissions_rel", "members"])
    return group


async def get_group(db: AsyncSession, group_id: uuid.UUID) -> Group | None:
    result = await db.execute(
        select(Group).where(Group.id == group_id).options(selectinload(Group.permissions_rel))
    )
    return result.scalar_one_or_none()


async def list_groups(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
    include_inactive: bool = False,
) -> tuple[list[Group], int]:
    query = select(Group).options(selectinload(Group.permissions_rel))
    if not include_inactive:
        query = query.where(Group.is_active == True)  # noqa: E712

    count_query = select(Group.id)
    if not include_inactive:
        count_query = count_query.where(Group.is_active == True)  # noqa: E712
    count_result = await db.execute(count_query)
    total = len(count_result.all())

    query = query.offset(offset).limit(limit).order_by(Group.name)
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_group(
    db: AsyncSession,
    group_id: uuid.UUID,
    **kwargs,
) -> Group | None:
    group = await db.get(Group, group_id)
    if not group:
        return None
    for key, value in kwargs.items():
        if hasattr(group, key) and value is not None:
            setattr(group, key, value)
    await db.commit()
    await db.refresh(group, attribute_names=["permissions_rel"])
    return group


async def delete_group(db: AsyncSession, group_id: uuid.UUID) -> bool:
    group = await db.get(Group, group_id)
    if not group:
        return False
    group.is_active = False
    await db.commit()
    return True


# --- Permission assignment ---


async def assign_permission(
    db: AsyncSession,
    group_id: uuid.UUID,
    permission: str,
) -> GroupPermission:
    gp = GroupPermission(group_id=group_id, permission=permission)
    db.add(gp)
    await db.commit()
    await db.refresh(gp)
    return gp


async def unassign_permission(
    db: AsyncSession,
    group_id: uuid.UUID,
    permission: str,
) -> bool:
    result = await db.execute(
        select(GroupPermission).where(
            GroupPermission.group_id == group_id, GroupPermission.permission == permission
        )
    )
    gp = result.scalar_one_or_none()
    if not gp:
        return False
    await db.delete(gp)
    await db.commit()
    return True


# --- Membership ---


async def add_member(
    db: AsyncSession,
    group_id: uuid.UUID,
    user_id: uuid.UUID,
) -> UserGroup:
    ug = UserGroup(user_id=user_id, group_id=group_id)
    db.add(ug)
    await db.commit()
    await db.refresh(ug)
    return ug


async def remove_member(
    db: AsyncSession,
    group_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    result = await db.execute(
        select(UserGroup).where(
            UserGroup.group_id == group_id, UserGroup.user_id == user_id
        )
    )
    ug = result.scalar_one_or_none()
    if not ug:
        return False
    await db.delete(ug)
    await db.commit()
    return True


async def list_user_groups(
    db: AsyncSession,
    user_id: uuid.UUID,
) -> list[Group]:
    result = await db.execute(
        select(Group)
        .join(UserGroup, UserGroup.group_id == Group.id)
        .where(UserGroup.user_id == user_id, Group.is_active == True)  # noqa: E712
        .options(selectinload(Group.permissions_rel))
    )
    return list(result.scalars().all())
