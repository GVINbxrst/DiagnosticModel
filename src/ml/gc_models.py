"""Очистка старых версий артефактов аномалий.

Удаляем старые файлы в models/anomaly_detection/ кроме каталога latest, сохраняя N последних по времени.
Интеграция: может вызываться из периодической задачи или вручную.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import List
from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

KEEP_VERSIONS = 5


def _parse_timestamp(name: str) -> datetime | None:
    # Ищем шаблон  YYYYMMDDTHHMMSSZ в имени файла
    import re
    m = re.search(r"(20\d{6}T\d{6}Z)", name)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")
    except Exception:  # pragma: no cover
        return None


def gc_anomaly_latest(keep: int = KEEP_VERSIONS) -> dict:
    st = get_settings()
    base = st.models_path / 'anomaly_detection'
    latest = base / 'latest'
    if not latest.exists():
        return {"status": "skipped", "reason": "latest_missing"}
    # Собираем файлы с timestamp в имени (pkl/json)
    candidates: List[Path] = [p for p in latest.iterdir() if p.is_file() and any(p.suffix == s for s in ('.pkl', '.json'))]
    with_ts = []
    for p in candidates:
        dt = _parse_timestamp(p.name)
        if dt:
            with_ts.append((dt, p))
    if not with_ts:
        return {"status": "skipped", "reason": "no_timestamp_files"}
    # Сортируем по времени убыв.
    with_ts.sort(key=lambda x: x[0], reverse=True)
    to_remove = with_ts[keep:]
    removed = []
    for _, path in to_remove:
        try:
            path.unlink()
            removed.append(path.name)
        except Exception as e:  # pragma: no cover
            logger.warning(f"GC remove fail {path}: {e}")
    return {"status": "ok", "removed": removed, "kept": keep}

__all__ = ["gc_anomaly_latest", "KEEP_VERSIONS"]
