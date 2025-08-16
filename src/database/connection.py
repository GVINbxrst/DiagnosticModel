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
import os, sys

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy import text

from src.config.settings import get_settings
from functools import wraps
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Подмена URL для тестового окружения если не переопределён пользователем.
db_url = settings.DATABASE_URL
if ('PYTEST_CURRENT_TEST' in os.environ or 'pytest' in sys.modules) and settings.APP_ENVIRONMENT != 'production':
    # Авто-подмена дефолтного Postgres на file-based SQLite (устойчиво между коннектами)
    if db_url.startswith('postgresql') and 'diagmod_user:diagmod_password' in db_url:
        db_url = 'sqlite+aiosqlite:///./test_db.sqlite'
        logger.info('Авто-подмена DATABASE_URL на file SQLite test_db.sqlite для тестов (вместо внешнего Postgres)')
    # Если явно указали :memory:, заменим на файл чтобы сохранить схему между соединениями aiosqlite
    elif db_url.endswith(':///:memory:'):
        db_url = 'sqlite+aiosqlite:///./test_db.sqlite'
        logger.info('Заменён sqlite in-memory на файл test_db.sqlite для устойчивости между соединениями')

# Engine согласно контракту. Используем NullPool чтобы избежать зависаний в тестах.
engine = create_async_engine(
    db_url,
    future=True,
    echo=getattr(settings, 'APP_DEBUG', False),
    poolclass=NullPool,
)

_SCHEMA_READY = False

async def _ensure_schema():  # pragma: no cover - инфраструктурный слой
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    # Авто-создание схемы только для sqlite тестовой БД
    if db_url.startswith('sqlite'):  # не трогаем Postgres
        try:
            from src.database.models import Base  # локальный импорт чтобы избежать циклов
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            _SCHEMA_READY = True
            logger.info('Инициализирована тестовая схема БД (SQLite)')
        except Exception:  # pragma: no cover
            logger.exception('Не удалось создать схему БД')


# Session maker согласно контракту.
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Асинхронный контекст для работы с БД с авто commit/rollback."""
    await _ensure_schema()
    async with async_session_maker() as session:
    # UUID coercion monkeypatch removed (handled by UniversalUUID type)
        try:
            yield session
            await session.commit()
        except Exception:
            try:
                await session.rollback()
            except Exception:
                logger.debug("Rollback failed", exc_info=True)
            raise

async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency variant (yield session)"""
    await _ensure_schema()
    async with async_session_maker() as session:
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
    'engine', 'async_session_maker', 'get_async_session', 'db_session', 'check_connection'
]
