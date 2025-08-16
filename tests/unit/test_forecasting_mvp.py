import pytest
import numpy as np
from uuid import uuid4
from datetime import datetime, timedelta
from sqlalchemy import text

from src.ml.forecasting import forecast_rms, InsufficientDataError
from src.database.connection import engine
from src.database.models import Base

@pytest.mark.asyncio
async def test_forecast_rms_mvp():
    equipment_id = uuid4()
    async with engine.begin() as conn:
        # Полное пересоздание схемы (drop + create) чтобы структура точно соответствовала моделям
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        # Связанный raw_signal (минимально требуемые поля)
        # Создаём оборудование для корректного внешнего ключа
        await conn.execute(text(
            "INSERT INTO equipment (id, equipment_id, name, type, status, created_at, updated_at) "
            "VALUES (:id,:code,:name,:type,:status,:ts,:ts)"
        ), {
            # SQLite драйвер не умеет биндинг python UUID напрямую -> приводим к str
            'id': str(equipment_id),
            'code': f'EQ-{str(equipment_id)[:8]}',
            'name': 'Test Motor',
            'type': 'induction_motor',
            'status': 'inactive',
            'ts': datetime.utcnow()
        })
        raw_signal_id = str(uuid4())
        await conn.execute(text(
            "INSERT INTO raw_signals (id, equipment_id, recorded_at, processing_status, processed, sample_rate_hz, samples_count, created_at, updated_at) "
            "VALUES (:id,:eq,:ts,'PENDING',0,:sr,:cnt,:ts,:ts)"
        ), {
            'id': raw_signal_id,           # строковое представление UUID
            'eq': str(equipment_id),       # приведение для sqlite
            'ts': datetime.utcnow(),
            'sr': 25600,
            'cnt': 1000
        })
        base_ts = datetime.utcnow() - timedelta(hours=150)
        for i in range(150):
            ts = base_ts + timedelta(hours=i)
            rms = 10 + 0.01 * i + np.random.normal(0, 0.2)
            await conn.execute(text("""
                INSERT INTO features (
                    id, raw_id, window_start, window_end,
                    rms_a, crest_a, kurt_a, skew_a, mean_a, std_a, min_a, max_a,
                    created_at, updated_at,
                    rms_b, rms_c, crest_b, crest_c, kurt_b, kurt_c,
                    skew_b, skew_c, mean_b, mean_c, std_b, std_c,
                    min_b, min_c, max_b, max_c, fft_spectrum, extra
                ) VALUES (
                    :id,:rid,:ws,:we,
                    :rms,1.0,0.0,0.0,:rms,0.1,:rmin,:rmax,
                    :ts,:ts,
                    NULL,NULL,NULL,NULL,NULL,NULL,
                    NULL,NULL,NULL,NULL,NULL,NULL,
                    NULL,NULL,NULL,NULL,NULL,NULL
                )
            """), {
                'id': str(uuid4()),
                'rid': raw_signal_id,
                'ws': ts,
                'we': ts + timedelta(minutes=1),
                'rms': float(rms),
                'rmin': float(rms) - 0.5,
                'rmax': float(rms) + 0.5,
                'ts': datetime.utcnow()
            })
    result = await forecast_rms(equipment_id, n_steps=12)
    assert 'forecast' in result and len(result['forecast']) == 12
    assert 'probability_over_threshold' in result
