"""Кеширование предсказаний с использованием Redis (если доступен) или fallback на SQLite таблицу prediction_cache."""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.utils.logger import get_logger
from src.database.connection import get_async_session

logger = get_logger(__name__)

_redis_client = None
_redis_init_attempted = False

async def _get_redis():
    global _redis_client, _redis_init_attempted
    if _redis_client or _redis_init_attempted:
        return _redis_client
    _redis_init_attempted = True
    try:
        import redis.asyncio as redis  # type: ignore
        url = get_settings().REDIS_URL
        _redis_client = redis.from_url(url, encoding="utf-8", decode_responses=True)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Redis unavailable, fallback to DB cache: {e}")
        _redis_client = None
    return _redis_client

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at INTEGER
);
"""

async def _ensure_sqlite_table(session: AsyncSession):
    await session.execute(text(CREATE_TABLE_SQL))
    await session.commit()

async def cache_prediction(cache_key: str, result: Any, expire_seconds: Optional[int] = None):
    expire = expire_seconds or get_settings().CACHE_EXPIRE_SECONDS
    payload = json.dumps({"data": result, "ts": int(time.time())})
    redis_client = await _get_redis()
    if redis_client:
        try:
            await redis_client.set(cache_key, payload, ex=expire)
            logger.debug(f"cache:set redis {cache_key}")
            return
        except Exception as e:  # pragma: no cover
            logger.warning(f"Redis set failed, fallback DB: {e}")
    # Fallback DB
    async with get_async_session() as session:
        await _ensure_sqlite_table(session)
        expires_at = int(time.time()) + expire
        await session.execute(text("REPLACE INTO prediction_cache(key,value,expires_at) VALUES (:k,:v,:e)"), {"k": cache_key, "v": payload, "e": expires_at})
        await session.commit()
        logger.debug(f"cache:set db {cache_key}")

async def get_cached_prediction(cache_key: str) -> Optional[Any]:
    redis_client = await _get_redis()
    if redis_client:
        try:
            raw = await redis_client.get(cache_key)
            if raw:
                logger.debug(f"cache:hit redis {cache_key}")
                try:
                    return json.loads(raw)["data"]
                except Exception:
                    return None
            logger.debug(f"cache:miss redis {cache_key}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Redis get failed: {e}")
    # Fallback DB
    async with get_async_session() as session:
        await _ensure_sqlite_table(session)
        now = int(time.time())
        res = await session.execute(text("SELECT value, expires_at FROM prediction_cache WHERE key=:k"), {"k": cache_key})
        row = res.first()
        if not row:
            logger.debug(f"cache:miss db {cache_key}")
            return None
        value, expires_at = row
        if expires_at is not None and now > int(expires_at):
            await session.execute(text("DELETE FROM prediction_cache WHERE key=:k"), {"k": cache_key})
            await session.commit()
            logger.debug(f"cache:expired db {cache_key}")
            return None
        logger.debug(f"cache:hit db {cache_key}")
        try:
            return json.loads(value)["data"]
        except Exception:
            return None

__all__ = ["cache_prediction", "get_cached_prediction"]
