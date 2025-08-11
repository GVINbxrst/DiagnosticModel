import asyncio
import types
import pytest
from sqlalchemy.exc import OperationalError

from src.database import connection as db_conn


@pytest.mark.asyncio
async def test_check_connection_success(monkeypatch):
    class DummyConn:
        async def execute(self, *a, **kw):
            return types.SimpleNamespace()
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class DummyEngine:
        def connect(self):
            return DummyConn()
    monkeypatch.setattr(db_conn, 'engine', DummyEngine())
    # reuse function
    assert await db_conn.check_connection() is True


@pytest.mark.asyncio
async def test_check_connection_failure(monkeypatch):
    class DummyConnBad:
        async def execute(self, *a, **kw):
            raise OperationalError('x','y','z')
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
    class DummyEngineBad:
        def connect(self):
            return DummyConnBad()
    monkeypatch.setattr(db_conn, 'engine', DummyEngineBad())
    ok = await db_conn.check_connection()
    assert ok is False
