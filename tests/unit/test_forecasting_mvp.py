import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta
from sqlalchemy import text

from src.ml.forecasting import forecast_rms, InsufficientDataError
from src.database.connection import engine

DDL_FEATURE = """
CREATE TABLE IF NOT EXISTS feature (
    id UUID PRIMARY KEY,
    raw_id UUID,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    rms_a DOUBLE PRECISION,
    rms_b DOUBLE PRECISION,
    rms_c DOUBLE PRECISION
);
CREATE TABLE IF NOT EXISTS raw_signal (
    id UUID PRIMARY KEY,
    equipment_id UUID
);
"""

@pytest.mark.asyncio
async def test_forecast_rms_mvp():
    equipment_id = uuid4()
    async with engine.begin() as conn:
        await conn.execute(text(DDL_FEATURE))
        # Связанные raw_signal
        await conn.execute(text("INSERT INTO raw_signal (id, equipment_id) VALUES (:id,:eq)"),{"id":str(uuid4()),'eq':str(equipment_id)})
        base_ts = datetime.utcnow() - timedelta(hours=150)
        # 150 точек (часовых) с небольшим трендом
        for i in range(150):
            ts = base_ts + timedelta(hours=i)
            rms = 10 + 0.01*i + np.random.normal(0,0.2)
            await conn.execute(text("INSERT INTO feature (id, raw_id, window_start, window_end, rms_a) VALUES (:id,:rid,:ws,:we,:rms)"),{
                'id': str(uuid4()),
                'rid': str(uuid4()),
                'ws': ts,
                'we': ts + timedelta(minutes=1),
                'rms': float(rms)
            })
    result = await forecast_rms(equipment_id, n_steps=12)
    assert 'forecast' in result and len(result['forecast']) == 12
    assert 'probability_over_threshold' in result
