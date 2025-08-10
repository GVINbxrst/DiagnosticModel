#!/usr/bin/env python3
"""
HTTP сервер для экспорта Prometheus метрик Worker
Запускается как отдельный процесс для предоставления метрик по HTTP
"""
import os
import sys
import time
import signal
import logging
from pathlib import Path
from prometheus_client import start_http_server, generate_latest, CONTENT_TYPE_LATEST
from flask import Flask, Response

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import get_all_metrics
from src.config.settings import get_settings

# Настройка логирования
setup_logging()
logger = get_logger(__name__)

# Настройки
settings = get_settings()
METRICS_PORT = int(os.getenv('WORKER_METRICS_PORT', 8002))
METRICS_HOST = os.getenv('WORKER_METRICS_HOST', '0.0.0.0')

# Flask приложение для метрик
app = Flask(__name__)


@app.route('/metrics')
def metrics():
    """Endpoint для Prometheus метрик"""
    try:
        metrics_data = get_all_metrics()
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {e}")
        return Response("Error getting metrics", status=500)


@app.route('/health')
def health():
    """Health check для Worker"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "diagmod-worker-metrics",
        "version": "1.0.0"
    }


@app.route('/ready')
def ready():
    """Readiness probe для Kubernetes"""
    return {
        "status": "ready",
        "timestamp": time.time()
    }


def signal_handler(signum, frame):
    """Обработчик сигналов для graceful shutdown"""
    logger.info(f"Получен сигнал {signum}, завершение работы...")
    sys.exit(0)


def main():
    """Основная функция запуска сервера метрик"""
    # Регистрируем обработчики сигналов
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"🚀 Запуск HTTP сервера метрик Worker на {METRICS_HOST}:{METRICS_PORT}")

    try:
        # Запускаем Flask сервер
        app.run(
            host=METRICS_HOST,
            port=METRICS_PORT,
            debug=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"❌ Ошибка запуска сервера метрик: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
