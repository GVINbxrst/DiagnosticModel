"""Утилиты для кеширования и сериализации ML моделей.

Функции:
 - save_model(obj, name): сохраняет объект под именем name в каталоге models
 - load_model(name): загружает объект из models

Учитывает настройку USE_MODEL_CACHE: если False, сохранение пропускается, загрузка возвращает None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Dict
import joblib

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INMEM_CACHE: Dict[str, Any] = {}


def _model_path(name: str) -> Path:
    st = get_settings()
    base = st.models_path
    path = Path(base) / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_model(obj: Any, name: str) -> Optional[Path]:
    st = get_settings()
    if not getattr(st, 'USE_MODEL_CACHE', True):
        logger.debug(f"Model cache disabled, skip saving '{name}'.")
        return None
    path = _model_path(name)
    try:
        joblib.dump(obj, path)
        _INMEM_CACHE[name] = obj  # сразу кладём в память
        logger.info(f"Saved model '{name}' -> {path}")
        return path
    except Exception as e:  # pragma: no cover
        logger.error(f"Failed saving model '{name}': {e}")
        return None


def load_model(name: str) -> Any | None:
    st = get_settings()
    if not getattr(st, 'USE_MODEL_CACHE', True):
        return None
    # In-memory fast path
    if name in _INMEM_CACHE:
        return _INMEM_CACHE[name]
    path = _model_path(name)
    if not path.exists():
        return None
    try:
        obj = joblib.load(path)
        _INMEM_CACHE[name] = obj
        logger.debug(f"Loaded model '{name}' from cache")
        return obj
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed loading model '{name}': {e}")
        return None

def warmup_models(names: list[str]) -> dict:
    """Загружает список моделей в память (тихо пропускает отсутствующие)."""
    loaded = {}
    for n in names:
        obj = load_model(n)
        if obj is not None:
            loaded[n] = True
    return loaded

__all__ = ["save_model", "load_model", "warmup_models"]
