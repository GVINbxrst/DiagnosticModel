from fastapi import FastAPI
from src.api.middleware.metrics import PrometheusMiddleware
from src.api.routes import monitoring
from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import metrics_collector

# Настр��йка логирования при запуске
setup_logging()
logger = get_logger(__name__)

app = FastAPI()

# Добавление middleware для метрик
app.add_middleware(PrometheusMiddleware)

# Добавление маршрутов мониторинга
app.include_router(
    monitoring.router,
    tags=["monitoring"],
    responses={404: {"description": "Not found"}}
)

@app.on_event("startup")
async def startup_event():
    """События при запуске приложения"""
    logger.info("🚀 Запуск DiagMod API")

    # Инициализация метрик
    metrics_collector.update_system_metrics()
    logger.info("📊 Система метрик инициализирована")

    # Здесь можно инициализировать дополнительные ресурсы при старте


@app.on_event("shutdown")
async def shutdown_event():
    """События при завершении работы приложения"""
    logger.info("🛑 Завершение работы DiagMod API")

    # Здесь можно освободить ресурсы при завершении
