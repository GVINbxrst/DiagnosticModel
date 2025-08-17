#!/usr/bin/env python3
"""HTTP сервер экспорта Prometheus метрик worker.

Изменения:
1. Удалены неиспользуемые импорты (CONTENT_TYPE_LATEST, generate_latest) и глобальные HOST/PORT.
2. Исключён sys.path.insert (предполагается установка проекта как пакета). 
3. Добавлены type hints для всех функций.
4. Добавлен argparse (опции: --host, --port, --exit-timeout) и приоритет конфигурации: CLI > ENV > settings > default.
5. Вынесен запуск сервера в функцию run_server(host, port).
6. Реализован graceful shutdown через threading.Event и обработчики SIGINT/SIGTERM (без прямого sys.exit в сигнале).
7. Обработана ошибка занятого порта (OSError) с логированием и кодом выхода 1.
8. Явный вызов get_all_metrics() внутри try/except с логированием ошибок и продолжением.
9. Цикл ожидания: while not stop_event.is_set(): sleep(2) (без пустого while True / KeyboardInterrupt).
10. Добавлено логирование URL и источника конфигурации (source=cli/env/settings/default).
11. Функция main(argv) -> int для тестируемости; в блоке запуска используется SystemExit(main(...)).
12. Совместимость с prometheus_client.start_http_server сохранена; без Flask и лишних зависимостей.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from typing import Iterable, List, Optional, Tuple

from prometheus_client import start_http_server

from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import get_all_metrics
from src.config.settings import get_settings

setup_logging()
logger = get_logger(__name__)

# Глобальное событие остановки (используется между обработчиком сигналов и основным циклом)
stop_event = threading.Event()


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    """Парсинг аргументов командной строки.

    Параметры:
        argv: список аргументов без имени скрипта (None -> использовать []).
    """
    parser = argparse.ArgumentParser(description="Worker metrics server")
    parser.add_argument("--host", help="Адрес прослушивания")
    parser.add_argument("--port", type=int, help="Порт прослушивания")
    parser.add_argument(
        "--exit-timeout",
        type=float,
        default=5.0,
        help="Максимальное время на корректное завершение (сек.)",
    )
    return parser.parse_args(argv or [])


def resolve_host_port(args: argparse.Namespace) -> Tuple[str, int, str]:
    """Определить host/port с указанием источника.

    Приоритет: CLI > ENV > settings > default.
    Возвращает (host, port, source_label).
    """
    settings = get_settings()

    source_parts: list[str] = []

    # HOST
    if args.host:
        host = args.host
        source_parts.append("host:cli")
    elif (env_host := os.getenv("WORKER_METRICS_HOST")):
        host = env_host
        source_parts.append("host:env")
    elif getattr(settings, "metrics_host", None):
        host = getattr(settings, "metrics_host")
        source_parts.append("host:settings")
    else:
        host = "0.0.0.0"
        source_parts.append("host:default")

    # PORT
    if args.port is not None:
        port = args.port
        source_parts.append("port:cli")
    elif (env_port := os.getenv("WORKER_METRICS_PORT")):
        try:
            port = int(env_port)
            source_parts.append("port:env")
        except ValueError:
            logger.warning("Некорректное значение WORKER_METRICS_PORT=%r, используем default", env_port)
            port = 8002
            source_parts.append("port:default")
    elif getattr(settings, "metrics_port", None):
        port = int(getattr(settings, "metrics_port"))
        source_parts.append("port:settings")
    else:
        port = 8002
        source_parts.append("port:default")

    return host, port, ",".join(source_parts)


def _signal_handler(signum, frame) -> None:  # type: ignore[override]
    """Обработчик сигналов: устанавливает событие остановки."""
    logger.info("Получен сигнал %s. Инициируем остановку...", signum)
    stop_event.set()


def run_server(host: str, port: int) -> bool:
    """Запуск HTTP сервера метрик.

    Возвращает True при успешном старте, False при ошибке порта.
    """
    logger.info("Регистрация метрик...")
    try:
        get_all_metrics()
    except Exception as exc:  # noqa: BLE001 - хотим логировать любую проблему и продолжить
        logger.warning("Не удалось полностью инициализировать метрики: %s", exc, exc_info=True)

    try:
        start_http_server(port, addr=host)
    except OSError as exc:  # Порт занят или нет прав
        logger.error("Не удалось запустить сервер метрик на %s:%s: %s", host, port, exc)
        return False

    return True


def main(argv: Optional[List[str]] = None) -> int:
    """Точка входа приложения.

    Возвращает код выхода (0 при успешной работе, 1 при ошибке старта).
    """
    args = parse_args(argv)
    host, port, source_label = resolve_host_port(args)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    if not run_server(host, port):
        return 1

    logger.info(
        "Метрики доступны: http://%s:%s/ (Prometheus формат) [config_sources=%s]",
        host,
        port,
        source_label,
    )
    logger.info("Ожидание сигналов для завершения (Ctrl+C для остановки)...")

    while not stop_event.is_set():
        # Здесь можно добавлять периодическое обновление пользовательских метрик
        time.sleep(2)

    logger.info("Завершение. Ожидание фоновых задач (%.1fs)...", args.exit_timeout)
    # Если появятся фоновые потоки: здесь join(...)
    return 0


if __name__ == "__main__":  # pragma: no cover - прямой запуск
    raise SystemExit(main(sys.argv[1:]))
