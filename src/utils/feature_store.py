"""Feature Store абстракция для быстрых выборок недавних агрегатов признаков.

Поддерживаются два бэкенда (управляется SETTINGS.FEATURE_STORE_BACKEND):
 - db (по умолчанию): обращается напрямую к таблицам Feature / HourlyFeatureSummary.
 - redis: хранит последние N метрик (RMS) в Redis для каждого оборудования.

Текущая реализация фокусируется на хранении последовательности (timestamp, rms_mean)
для ускорения построения графиков и расчёта краткосрочных рисков без тяжёлых JOIN.

Функции:
 - store_rms(equipment_id, ts, rms_mean)
 - get_recent_rms(equipment_id, limit)
 - get_range_rms(equipment_id, start, end)

Примечание: Для backend="db" операции фактически проксируют к HourlyFeatureSummary
 (если есть точное попадание по часу) либо к агрегату Feature, чтобы не дублировать данные.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict
from uuid import UUID

from sqlalchemy import select, func

from src.config.settings import get_settings
from src.utils.metrics import increment_counter
from src.database.connection import get_async_session
from src.database.models import Feature, RawSignal, HourlyFeatureSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)

_store_instance = None  # singleton


class BaseFeatureStore:
    async def store_rms(self, equipment_id: UUID, ts: datetime, rms_mean: float):  # pragma: no cover - интерфейс
        raise NotImplementedError

    async def get_recent_rms(self, equipment_id: UUID, limit: int = 100) -> List[Dict]:  # pragma: no cover
        raise NotImplementedError

    async def get_range_rms(self, equipment_id: UUID, start: datetime, end: datetime) -> List[Dict]:  # pragma: no cover
        raise NotImplementedError


class DBFeatureStore(BaseFeatureStore):
    """Бэкенд без отдельного кеша: читаем из HourlyFeatureSummary либо агрегируем Feature."""

    async def store_rms(self, equipment_id: UUID, ts: datetime, rms_mean: float):
        # Для DB backend отдельного хранения не требуется (у нас есть Feature + HourlyFeatureSummary).
        return None

    async def get_recent_rms(self, equipment_id: UUID, limit: int = 100) -> List[Dict]:
        async with get_async_session() as session:  # type: ignore
            # Сначала пытаемся взять почасовые агрегаты (они компактны)
            q_hours = (
                select(HourlyFeatureSummary)
                .where(HourlyFeatureSummary.equipment_id == equipment_id)
                .order_by(HourlyFeatureSummary.hour_start.desc())
                .limit(limit)
            )
            res = await session.execute(q_hours)
            rows = res.scalars().all()
            if rows:
                increment_counter('feature_store_hit_total', {'backend': 'db'})
                return [
                    {
                        'timestamp': r.hour_start.isoformat(),
                        'rms_mean': float(r.rms_mean) if r.rms_mean is not None else None,
                        'samples': r.samples
                    } for r in rows
                ][::-1]
            # Fallback: агрегируем оконные признаки Feature
            q = (
                select(
                    Feature.window_start.label('ts'),
                    func.avg(
                        (func.coalesce(Feature.rms_a, 0) + func.coalesce(Feature.rms_b, 0) + func.coalesce(Feature.rms_c, 0)) / 3.0
                    ).label('rms_mean')
                )
                .join(RawSignal)
                .where(RawSignal.equipment_id == equipment_id)
                .group_by(Feature.window_start)
                .order_by(Feature.window_start.desc())
                .limit(limit)
            )
            res = await session.execute(q)
            out = []
            for ts, rms_mean in res.fetchall():
                out.append({'timestamp': ts.isoformat(), 'rms_mean': float(rms_mean) if rms_mean is not None else None})
            if out:
                increment_counter('feature_store_hit_total', {'backend': 'db'})
            else:
                increment_counter('feature_store_miss_total', {'backend': 'db', 'reason': 'empty'})
            return out[::-1]

    async def get_range_rms(self, equipment_id: UUID, start: datetime, end: datetime) -> List[Dict]:
        async with get_async_session() as session:  # type: ignore
            q = (
                select(HourlyFeatureSummary)
                .where(
                    HourlyFeatureSummary.equipment_id == equipment_id,
                    HourlyFeatureSummary.hour_start >= start,
                    HourlyFeatureSummary.hour_start <= end
                )
                .order_by(HourlyFeatureSummary.hour_start.asc())
            )
            res = await session.execute(q)
            rows = res.scalars().all()
            return [
                {
                    'timestamp': r.hour_start.isoformat(),
                    'rms_mean': float(r.rms_mean) if r.rms_mean is not None else None,
                    'samples': r.samples
                } for r in rows
            ]


class RedisFeatureStore(BaseFeatureStore):
    """Хранит последние N точек в Redis списке. Ключ: feature_store:{equipment_id}"""

    KEY_PREFIX = "feature_store"
    # MAX_POINTS теперь берём из настроек динамически

    def __init__(self):
        self._client = None

    async def _client_or_none(self):
        if self._client is not None:
            return self._client
        try:  # pragma: no cover - внешняя инфраструктура
            import redis.asyncio as redis  # type: ignore
            url = get_settings().REDIS_URL
            self._client = redis.from_url(url, encoding="utf-8", decode_responses=True)
        except Exception as e:
            logger.warning(f"RedisFeatureStore недоступен: {e}")
            self._client = None
        return self._client

    async def store_rms(self, equipment_id: UUID, ts: datetime, rms_mean: float):
        cli = await self._client_or_none()
        if not cli:
            return
        key = f"{self.KEY_PREFIX}:{equipment_id}"
        try:
            import json
            value = json.dumps({'ts': ts.isoformat(), 'rms_mean': rms_mean})
            # LPUSH + LTRIM для ограничения размера
            settings = get_settings()
            max_points = getattr(settings, 'MAX_FEATURE_STORE_POINTS', 2000)
            await cli.lpush(key, value)
            await cli.ltrim(key, 0, max_points - 1)
        except Exception as e:  # pragma: no cover
            logger.warning(f"RedisFeatureStore.store_rms ошибка: {e}")

    async def get_recent_rms(self, equipment_id: UUID, limit: int = 100) -> List[Dict]:
        cli = await self._client_or_none()
        if not cli:
            # Fallback: DB
            increment_counter('feature_store_miss_total', {'backend': 'redis', 'reason': 'no_client'})
            return await DBFeatureStore().get_recent_rms(equipment_id, limit)
        key = f"{self.KEY_PREFIX}:{equipment_id}"
        try:
            values = await cli.lrange(key, 0, limit - 1)
            import json
            out = [json.loads(v) for v in values]
            if not out:
                increment_counter('feature_store_miss_total', {'backend': 'redis', 'reason': 'empty'})
            else:
                increment_counter('feature_store_hit_total', {'backend': 'redis'})
            return list(reversed(out))  # храним в обратном порядке (LPUSH)
        except Exception as e:  # pragma: no cover
            logger.warning(f"RedisFeatureStore.get_recent_rms ошибка: {e}")
            increment_counter('feature_store_miss_total', {'backend': 'redis', 'reason': 'exception'})
            return []

    async def get_range_rms(self, equipment_id: UUID, start: datetime, end: datetime) -> List[Dict]:
        # Диапазон неэффективен в списке — используем recent и фильтруем
        settings = get_settings()
        max_points = getattr(settings, 'MAX_FEATURE_STORE_POINTS', 2000)
        recent = await self.get_recent_rms(equipment_id, max_points)
        # Исправляем ключ 'ts' -> 'timestamp' если DB fallback возвращал другие ключи
        normalized = []
        for r in recent:
            ts = r.get('ts') or r.get('timestamp')
            if ts is None:
                continue
            normalized.append({'timestamp': ts, 'rms_mean': r.get('rms_mean')})
        return [r for r in normalized if start.isoformat() <= r['timestamp'] <= end.isoformat()]  # type: ignore


async def get_feature_store() -> BaseFeatureStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    backend = get_settings().FEATURE_STORE_BACKEND.lower()
    if backend == 'redis':  # pragma: no branch - простое ветвление
        _store_instance = RedisFeatureStore()
    else:
        _store_instance = DBFeatureStore()
    logger.info(f"FeatureStore инициализирован: backend={backend}")
    return _store_instance

__all__ = [
    'get_feature_store',
    'BaseFeatureStore'
]
