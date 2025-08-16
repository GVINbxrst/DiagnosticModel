# (Де)сериализация float32 массивов в gzip bytes (центр. утилиты)
from __future__ import annotations

import gzip
from typing import Optional

import numpy as np


def dump_float32_array(arr: Optional[np.ndarray]) -> Optional[bytes]:
    # Сжать numpy массив в gzip bytes
    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)
    if arr.size == 0:
        return None
    return gzip.compress(arr.astype(np.float32).tobytes(), compresslevel=6)


def load_float32_array(b: Optional[bytes]) -> Optional[np.ndarray]:
    # Распаковать gzip bytes в float32 массив
    if not b:
        return None
    data = gzip.decompress(b)
    return np.frombuffer(data, dtype=np.float32)


__all__ = ["dump_float32_array", "load_float32_array"]
