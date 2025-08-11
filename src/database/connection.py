"""Database connection utilities (async SQLAlchemy v2 style).

Требования (контракт задачи):
 - engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool, future=True)
 - async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
 - get_async_session(): контекстный менеджер commit/rollback
 - check_connection(): вернуть True/False, логируя исключение

Минимально инвазивная реализация; параметры пула/echo берём из settings.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from src.config.settings import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Engine согласно контракту. Используем NullPool чтобы избежать зависаний в тестах.
engine = create_async_engine(
    settings.DATABASE_URL,
    future=True,
    echo=getattr(settings, 'APP_DEBUG', False),
    poolclass=NullPool,
)

# Session maker согласно контракту.
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронный контекст для работы с БД с авто commit/rollback."""
    async with async_session_maker() as session:  # type: AsyncSession
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                logger.debug("Rollback failed", exc_info=True)
            raise


async def check_connection() -> bool:
    """Проверить доступность соединения (SELECT 1)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("DB connection test failed")
        return False


__all__ = [
    'engine', 'async_session_maker', 'get_async_session', 'check_connection'
]
