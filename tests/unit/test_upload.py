"""Интеграционный тест /upload: полная цепочка без упрощений.

API -> CSVLoader -> БД (SQLite in-memory для теста) -> Celery (eager) -> FeatureExtractor -> статус RawSignal.

Важно: не упрощаем бизнес-логику, только конфигурируем окружение теста.
"""

import os
import asyncio
import importlib
import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

# Настраиваем окружение ДО импорта модулей, чтобы settings/engine подхватили значения
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("CELERY_TASK_EAGER_PROPAGATES", "true")
os.environ.setdefault("APP_ENVIRONMENT", "test")

from src.config.settings import get_settings
get_settings.cache_clear()  # сброс кэша чтобы перечитать переменные
settings = get_settings()

# Переинициализируем модуль соединения с БД под новый DATABASE_URL
from src import database as _db_pkg  # noqa
from src.database import connection as db_conn
importlib.reload(db_conn)  # пересоздаёт engine и async_session_maker
from src.database.connection import engine, get_async_session
from src.database.models import Base, RawSignal, Feature, ProcessingStatus

# Celery app (используем штатный, переключаем в eager режим)
from src.worker.config import celery_app
celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

from src.api.routes.upload import router

# ---------------- Fixtures ----------------

@pytest_asyncio.fixture(scope="session", autouse=True)
def _display_env():
    # Информационный вывод (можно логировать при необходимости)
    return {
        'db_url': settings.DATABASE_URL,
        'celery_eager': celery_app.conf.task_always_eager,
    }


@pytest_asyncio.fixture(scope="session")
async def initialized_db():
    # Создаём схемы
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


from typing import AsyncGenerator

@pytest_asyncio.fixture
async def session(initialized_db) -> AsyncGenerator[AsyncSession, None]:
    async with get_async_session() as s:
        yield s


@pytest_asyncio.fixture
async def app(session):
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_upload_success_full_pipeline(app, session: AsyncSession):
    csv_content = 'current_R,current_S,current_T\n1,2,3\n4,5,6\n7,8,9\n'
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("test.csv", csv_content, "text/csv")}
        resp = await ac.post("/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    raw_id = body['raw_id']
    assert body['status'] == 'queued'

    # Так как Celery eager, задача уже должна выполниться.
    # Проверяем статус RawSignal и наличие признаков.
    # Поллинг на случай если даже в eager_mode есть задержка commit в других async контекстах
    async def _fetch():
        async with get_async_session() as s:
            rs = await s.get(RawSignal, raw_id)
            feats = []
            if rs:
                feats = (await s.execute(select(Feature).where(Feature.raw_id == raw_id))).scalars().all()
            return rs, feats

    rs, feats = None, []
    for _ in range(5):
        rs, feats = await _fetch()
        if rs and rs.processing_status == ProcessingStatus.COMPLETED and feats:
            break
        await asyncio.sleep(0.05)

    assert rs is not None, "RawSignal не сохранён"
    assert rs.processing_status == ProcessingStatus.COMPLETED
    assert len(feats) > 0, "Признаки не извлечены"


@pytest.mark.asyncio
async def test_upload_bad_header(app):
    csv_content = 'bad,header,here\n1,2,3\n'
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        files = {"file": ("bad.csv", csv_content, "text/csv")}
        resp = await ac.post("/upload", files=files)
    assert resp.status_code == 422
    assert 'current_R' in resp.text
