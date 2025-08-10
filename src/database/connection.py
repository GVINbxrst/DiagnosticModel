"""
Подключение к базе данных для DiagMod
Асинхронные сессии SQLAlchemy для работы с PostgreSQL
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.config.settings import get_settings

# Настройки
settings = get_settings()

# Создание асинхронного движка
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_DEBUG,
    pool_pre_ping=True,
    pool_recycle=3600,
    poolclass=NullPool if settings.APP_ENVIRONMENT == "test" else None,
    connect_args={
        "server_settings": {
            "application_name": "diagmod_app",
        }
    }
)

# Фабрика сессий
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Получить асинхронную сессию базы данных

    Yields:
        AsyncSession: Сессия базы данных
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_async_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    Зависимость FastAPI для получения сессии базы данных

    Yields:
        AsyncSession: Сессия базы данных
    """
    async with get_async_session() as session:
        yield session


async def create_tables():
    """Создать все таблицы в базе данных"""
    from src.database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Удалить все таблицы из базы данных"""
    from src.database.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_connection():
    """Проверить подключение к базе данных"""
    try:
        async with get_async_session() as session:
            result = await session.execute("SELECT 1")
            return result.scalar() == 1
    except Exception:
        return False


if __name__ == "__main__":
    # Тест подключения
    async def test_connection():
        print("Тестируем подключение к базе данных...")

        if await check_connection():
            print("✓ Подключение успешно")
        else:
            print("✗ Ошибка подключения")

    asyncio.run(test_connection())
