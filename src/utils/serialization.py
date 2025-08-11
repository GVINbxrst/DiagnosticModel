"""Утилиты централизованной (де)сериализации числовых массивов.

Контракт:
- Храним массивы float32 в БД как gzip (level=6) сжатые сырые bytes (little-endian
  стандартный порядок numpy.tobytes()).
- Пустой или None -> None (ничего не сохраняем).

Использование только через эти функции – прямые обращения к gzip / np.frombuffer
для фазовых сигналов запрещены.
"""
from __future__ import annotations

import gzip
from typing import Optional

import numpy as np


def dump_float32_array(arr: Optional[np.ndarray]) -> Optional[bytes]:
    """Сжать numpy массив в gzip bytes.

    Args:
        arr: Массив (любой dtype) или None.

    Returns:
        gzip bytes или None если нет данных.
    """
    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    return gzip.compress(arr.astype(np.float32).tobytes(), compresslevel=6)


def load_float32_array(b: Optional[bytes]) -> Optional[np.ndarray]:
    """Распаковать gzip bytes в numpy float32.

    Args:
        b: gzip bytes или None.

    Returns:
        Массив float32 или None если вход пуст.
    """
    if not b:
        return None
    data = gzip.decompress(b)
    return np.frombuffer(data, dtype=np.float32)


__all__ = ["dump_float32_array", "load_float32_array"]
