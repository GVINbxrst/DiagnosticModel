"""Тест /upload: FastAPI + SQLite in-memory + Celery eager + полный пайплайн.

Проверяем:
 1. RawSignal создаётся
 2. processing_status -> COMPLETED
 3. Признаки (Feature) созданы
"""

import os
import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

os.environ.setdefault("APP_ENVIRONMENT", "test")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("CELERY_TASK_EAGER_PROPAGATES", "true")

DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

from src.database.models import Base, RawSignal, Feature, ProcessingStatus
from src.worker.config import celery_app
from src.api.routes.upload import router

celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = True

from contextlib import asynccontextmanager

@asynccontextmanager
async def override_get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session

# Monkeypatch заменяем только если явно нужно; здесь оставим оригинальный интерфейс get_async_session, но подменим engine в модуле.
import src.database.connection as conn_mod  # noqa
conn_mod.engine = engine  # type: ignore
conn_mod.async_session_maker = SessionLocal  # type: ignore
conn_mod.get_async_session = override_get_async_session  # type: ignore

@pytest_asyncio.fixture(scope="session", autouse=True)
async def _create_schema():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app

@pytest.mark.asyncio
async def test_upload_full_pipeline(app: FastAPI):
    csv_content = 'current_R,current_S,current_T\n1,2,3\n4,5,6\n7,8,9\n'
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/upload", files={"file": ("test.csv", csv_content, "text/csv")})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    raw_id = body['raw_id']

    async def fetch():
        async with override_get_async_session() as s:  # type: ignore
            rs = await s.get(RawSignal, raw_id)
            feats = []
            if rs:
                feats = (await s.execute(select(Feature).where(Feature.raw_id == raw_id))).scalars().all()
            return rs, feats

    rs, feats = None, []
    for _ in range(10):
        rs, feats = await fetch()
        if rs and rs.processing_status == ProcessingStatus.COMPLETED and feats:
            break
        await asyncio.sleep(0.05)

    assert rs is not None, "RawSignal не создан"
    assert rs.processing_status == ProcessingStatus.COMPLETED
    assert len(feats) > 0, "Нет признаков"
