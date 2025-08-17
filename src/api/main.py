from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.database.connection import get_async_session  # re-export for tests patching
from src.api.middleware.metrics import PrometheusMiddleware
from src.api.routes import monitoring, signals, upload, auth, anomalies, admin_security, pipeline_status
import inspect
try:
    import httpx
    if 'app' not in inspect.signature(httpx.AsyncClient.__init__).parameters:  # версия httpx>=0.28
        _orig_ac_init = httpx.AsyncClient.__init__
        def _patched_ac_init(self, *args, app=None, base_url='', **kwargs):  # type: ignore
            if app is not None and 'transport' not in kwargs:
                from httpx import ASGITransport
                kwargs['transport'] = ASGITransport(app=app)
            if base_url and 'base_url' not in kwargs:
                kwargs['base_url'] = base_url
            return _orig_ac_init(self, **kwargs)
        httpx.AsyncClient.__init__ = _patched_ac_init  # type: ignore
except Exception:  # pragma: no cover
    pass
from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import metrics_collector
from src.config.settings import get_settings

async def _maybe_run_startup_ingest():  # pragma: no cover - инфраструктура
    st = get_settings()
    if not st.INGEST_STARTUP_DIR:
        return
    import os, asyncio
    from scripts.e2e_pipeline import run_pipeline  # type: ignore
    path = st.INGEST_STARTUP_DIR
    if not os.path.isdir(path):
        logger.warning(f"Startup ingest: каталог не найден {path}")
        return
    logger.info(f"Startup ingest: начало обработки каталога {path}")
    try:
        await run_pipeline(path, None, st.INGEST_TRAIN_FULL, st.INGEST_FILE_PATTERN, st.INGEST_MAX_FILES)
        logger.info("Startup ingest завершён")
    except Exception as e:
        logger.warning(f"Startup ingest не удался: {e}")

# Настройка логирования при запуске
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover - оборачиваем стартап/шутдаун
    logger.info("🚀 Запуск DiagMod API")
    try:
        metrics_collector.update_system_metrics()
        logger.info("📊 Система метрик инициализирована")
    except Exception:  # не валим приложение из-за метрик
        logger.warning("Не удалось инициализировать метрики при старте", exc_info=True)
    # Автоинжест (опционально)
    await _maybe_run_startup_ingest()
    yield
    try:
        logger.info("🛑 Завершение работы DiagMod API")
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)

# Добавление middleware для метрик
app.add_middleware(PrometheusMiddleware)

# Корневой health/root
@app.get("/")
async def root():
    return {"status":"healthy","message":"DiagMod API работает корректно","version":"1.0.0"}
from sqlalchemy import text
from fastapi import HTTPException, status as http_status

@app.get("/health")
async def health_alias():
    """Универсальный health чек, используемый тестами.

    Требования тестов:
     - Возвращать 200 и тело со status=="healthy", а также ключ "checks" при успешном подключении к БД
     - При ошибке БД вернуть 503 и status=="unhealthy"
    """
    checks = {}
    try:
        async with get_async_session() as session:  # type: ignore
            await session.execute(text("SELECT 1"))
        checks["database"] = {"status": "healthy"}
        return {"status": "healthy", "service": "diagmod-api", "version": "1.0.0", "checks": checks}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, content={
            "status": "unhealthy", "service": "diagmod-api", "version": "1.0.0", "checks": checks
        })

# Подключение всех роутеров
app.include_router(monitoring.router, tags=["monitoring"], prefix="/monitoring", responses={404:{"description":"Not found"}})
app.include_router(signals.router, prefix="/api/v1", tags=["signals"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(anomalies.router, prefix="/api/v1", tags=["anomalies"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])  # версия API
app.include_router(auth.router, prefix="/auth", tags=["auth"])    # legacy совместимость для тестов
app.include_router(admin_security.router, prefix="/admin/security", tags=["admin-security"])  # административные security эндпоинты
app.include_router(pipeline_status.router, prefix="/api/v1", tags=["pipeline"])  # статус конвейера загрузки

## Удалены on_event startup/shutdown (заменены lifespan)
