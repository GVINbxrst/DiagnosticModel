#!/usr/bin/env python
"""Генерация простого аналитического отчёта по статистике БД.

Собирает агрегаты: кол-во RawSignal по статусам, Features, Predictions, аномалии.
"""
import asyncio
from collections import Counter
from sqlalchemy import select, func

from src.database.connection import get_async_session
from src.database.models import RawSignal, Feature, Prediction, ProcessingStatus

async def build_report():
    async with get_async_session() as session:
        # Статусы сырья
        res = await session.execute(select(RawSignal.processing_status, func.count()).group_by(RawSignal.processing_status))
        status_counts = {str(r[0]): r[1] for r in res.fetchall()}
        # Кол-во признаков
        feat_cnt = (await session.execute(select(func.count(Feature.id)))).scalar() or 0
        # Кол-во предсказаний
        pred_cnt = (await session.execute(select(func.count(Prediction.id)))).scalar() or 0
        # Аномалии
        anom_cnt = (await session.execute(select(func.count(Prediction.id)).where(Prediction.anomaly_detected == True))).scalar() or 0
    return {
        'raw_signal_status': status_counts,
        'features_total': int(feat_cnt),
        'predictions_total': int(pred_cnt),
        'anomalies_total': int(anom_cnt),
    }

if __name__ == '__main__':
    rep = asyncio.run(build_report())
    print("=== Analytics Report ===")
    for k,v in rep.items():
        print(f"{k}: {v}")
