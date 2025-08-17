"""Конфигурация стандартного logging для приложения.

Уровень:
 - development / test: INFO (можно поднять через settings.LOG_LEVEL / переменную окружения)
 - production: WARNING по умолчанию

Формат: время | уровень | модуль | сообщение
Логи в файл, если указан settings.LOG_FILE_PATH, иначе только консоль.
"""
from __future__ import annotations

import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings

_configured = False


def configure_logging(force: bool = False) -> None:
    global _configured
    if _configured and not force:
        return

    settings = get_settings()

    # Базовый уровень: prod -> WARNING, иначе settings.LOG_LEVEL (обычно INFO)
    base_level_name = settings.LOG_LEVEL
    if settings.is_production and base_level_name in ("DEBUG", "INFO"):
        base_level_name = "WARNING"
    level = getattr(logging, base_level_name.upper(), logging.INFO)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Чистим корневой логгер (idempotent reconfigure)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
    root.addHandler(console_handler)

    if settings.LOG_FILE_PATH:
        try:
            path = Path(settings.LOG_FILE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=datefmt))
            root.addHandler(file_handler)
        except Exception as e:  # pragma: no cover - деградация в stdout
            root.warning(f"Не удалось настроить файл логов: {e}")

    _configured = True


def get_logger(name: Optional[str] = None) -> Logger:
    configure_logging()
    return logging.getLogger(name or __name__)


__all__ = ["configure_logging", "get_logger"]
