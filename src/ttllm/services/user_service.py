"""User CRUD operations."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ttllm.core.password import hash_password
from ttllm.models.user import User


async def create_user(
    db: AsyncSession,
    name: str,
    email: str,
    password: str | None = None,
    identity_provider: str | None = None,
    external_id: str | None = None,
) -> User:
    user = User(
        name=name,
        email=email,
        password_hash=hash_password(password) if password else None,
        identity_provider=identity_provider,
        external_id=external_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await db.get(User, user_id)


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession,
    offset: int = 0,
    limit: int = 50,
    include_inactive: bool = False,
) -> tuple[list[User], int]:
    query = select(User)
    if not include_inactive:
        query = query.where(User.is_active == True)  # noqa: E712

    count_query = select(User.id)
    if not include_inactive:
        count_query = count_query.where(User.is_active == True)  # noqa: E712
    count_result = await db.execute(count_query)
    total = len(count_result.all())

    query = query.offset(offset).limit(limit).order_by(User.created_at.desc())
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def update_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    **kwargs,
) -> User | None:
    user = await db.get(User, user_id)
    if not user:
        return None

    # Handle password separately
    if "password" in kwargs:
        pw = kwargs.pop("password")
        if pw is not None:
            user.password_hash = hash_password(pw)

    for key, value in kwargs.items():
        if hasattr(user, key) and value is not None:
            setattr(user, key, value)
    await db.commit()
    await db.refresh(user)
    return user


async def deactivate_user(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await update_user(db, user_id, is_active=False)
