import time
import pytest
from src.utils import metrics as m


def test_increment_and_histogram():
    # Создаем временные метрики
    m.increment('api_requests_total', {'method':'GET','endpoint':'/x','status_code':'200','user_role':'test'})
    # Используем observe_latency декоратор
    @m.observe_latency('api_request_duration_seconds', {'method':'TEST','endpoint':'/unit'})
    def slow():
        time.sleep(0.01)
        return 42
    val = slow()
    assert val == 42


@pytest.mark.asyncio
async def test_async_latency():
    @m.observe_latency('api_request_duration_seconds', {'method':'ASYNC','endpoint':'/async'})
    async def work():
        await asyncio.sleep(0.005)
        return 'ok'
    import asyncio
    res = await work()
    assert res == 'ok'
