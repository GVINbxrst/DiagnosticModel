import io
import pytest
import os
os.environ['APP_ENVIRONMENT']='test'
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from fastapi import FastAPI

from src.data_processing.csv_loader import CSVLoader


@pytest.mark.asyncio
async def test_csv_loader_emits_batch_metrics(tmp_path, monkeypatch):
    # Подготовим небольшой CSV
    p = tmp_path / 'm.csv'
    p.write_text('current_R,current_S,current_T\n' + '\n'.join(['1,2,3' for _ in range(5)]))

    loader = CSVLoader(batch_size=3)

    # Мокаем БД
    with patch('src.data_processing.csv_loader.get_async_session') as mock_sess:
        s = AsyncMock()
        mock_sess.return_value.__aenter__.return_value = s
        # Файл не загружен ранее
        s.execute.return_value.scalar_one_or_none.return_value = None

        # Мокаем оборудование
        with patch('src.data_processing.csv_loader.find_equipment_by_filename') as mock_eq:
            eq = AsyncMock()
            eq.id = uuid4()
            mock_eq.return_value = eq

            # Перехватим вызовы метрик
            calls = {"rows": [], "dur": [], "points": []}
            with patch('src.data_processing.csv_loader.m.observe_histogram') as obs, \
                 patch('src.data_processing.csv_loader.m.increment_counter') as inc:
                await loader.load_csv_file(p)

                # Соберем аргументы
                for c in obs.call_args_list:
                    name = c.args[0]
                    if name == 'csv_batch_rows':
                        calls["rows"].append(c.args[1])
                    if name == 'csv_batch_duration_seconds':
                        calls["dur"].append(c.args[1])
                for c in inc.call_args_list:
                    if c.args[0] == 'data_points_processed_total':
                        calls["points"].append(c.kwargs.get('value', 0))

            # Проверим, что хотя бы один батч зафиксирован
            assert any(v >= 3.0 for v in calls["rows"]) or len(calls["rows"])>0
            # По точкам данных — инкременты присутствуют
            assert sum(calls["points"]) > 0
