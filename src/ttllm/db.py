from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ttllm.config import settings

engine = create_async_engine(
    settings.database.url,
    pool_size=settings.database.pool_size,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
