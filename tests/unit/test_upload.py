import io
import pytest
from fastapi.testclient import TestClient
from src.api.routes.upload import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

def test_upload_success(client, monkeypatch):
    # Мокаем loader и celery
    class DummyTask:
        id = 'task-123'
    async def fake_load_csv_file(path, **kwargs):
        return 'raw-uuid-1'
    def fake_process_raw(raw_id):
        return DummyTask()
    monkeypatch.setattr('src.data_processing.csv_loader.CSVLoader.load_csv_file', fake_load_csv_file)
    monkeypatch.setattr('src.worker.tasks.process_raw.delay', fake_process_raw)
    csv = 'current_R,current_S,current_T\n1,2,3\n4,5,6\n'
    resp = client.post('/upload', files={'file': ('test.csv', csv, 'text/csv')})
    assert resp.status_code == 200
    data = resp.json()
    assert 'raw_id' in data and 'task_id' in data and data['status'] == 'queued'

def test_upload_bad_header(client, monkeypatch):
    async def fake_load_csv_file(path, **kwargs):
        return 'raw-uuid-1'
    monkeypatch.setattr('src.data_processing.csv_loader.CSVLoader.load_csv_file', fake_load_csv_file)
    csv = 'bad,header,here\n1,2,3\n'
    resp = client.post('/upload', files={'file': ('test.csv', csv, 'text/csv')})
    assert resp.status_code == 422
    assert 'header' in resp.text or 'current_R' in resp.text
