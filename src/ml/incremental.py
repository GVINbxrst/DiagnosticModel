"""Инкрементальные модели аномалий.

Добавляем адаптер для river HalfSpaceTrees (потоковое обнаружение аномалий) и
скользящее переобучение IsolationForest как fallback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Iterable, Tuple
import numpy as np

try:  # river может отсутствовать
    from river import anomaly, drift
    RIVER_AVAILABLE = True
except Exception:  # pragma: no cover
    RIVER_AVAILABLE = False

from src.utils.logger import get_logger
logger = get_logger(__name__)

@dataclass
class IncrementalAnomalyResult:
    score: float
    is_anomaly: bool
    threshold: float

class HalfSpaceTreesIncremental:
    """Обёртка над river.anomaly.HalfSpaceTrees для пошаговой обработки.
    Используем score > threshold как индикатор аномалии.
    """
    def __init__(self, window_size: int = 250, n_trees: int = 25, height: int = 15, threshold: float = 0.7):
        if not RIVER_AVAILABLE:
            raise ImportError("river не установлен. Добавьте 'river' в зависимости для инкрементального режима.")
        self.model = anomaly.HalfSpaceTrees(window_size=window_size, n_trees=n_trees, height=height)
        self.threshold = threshold

    def learn_one(self, x: Dict[str, float]) -> IncrementalAnomalyResult:
        score = self.model.score_one(x)
        self.model.learn_one(x)
        return IncrementalAnomalyResult(score=score, is_anomaly=score > self.threshold, threshold=self.threshold)

    def predict_batch(self, rows: Iterable[Dict[str, float]]) -> List[IncrementalAnomalyResult]:
        results = []
        for r in rows:
            results.append(self.learn_one(r))
        return results

class SlidingIsolationForest:  # DEPRECATED: больше не используется; оставить до удаления в следующем релизе
    """(DEPRECATED) Псевдо-инкрементальный IsolationForest: накапливает окна и переобучается по буферу.
    Сохранён для обратной совместимости, но не вызывается.
    Будет удалён в последующих версиях.
    """
    def __init__(self, max_buffer: int = 5000, retrain_every: int = 500):
        from sklearn.ensemble import IsolationForest  # локальный импорт
        self.max_buffer = max_buffer
        self.retrain_every = retrain_every
        self.buffer: List[List[float]] = []
        self.model: Optional[IsolationForest] = None
        self._seen = 0

    def partial_fit(self, X: np.ndarray):
        # Добавляем в буфер
        for row in X:
            self.buffer.append(row.tolist())
        # Ограничиваем размер буфера
        if len(self.buffer) > self.max_buffer:
            self.buffer = self.buffer[-self.max_buffer:]
        self._seen += len(X)
        # Переобучаем периодически
        if self.model is None or self._seen >= self.retrain_every:
            from sklearn.ensemble import IsolationForest
            arr = np.array(self.buffer)
            self.model = IsolationForest(n_estimators=200, contamination='auto', n_jobs=-1, random_state=42)
            self.model.fit(arr)
            self._seen = 0

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.model:
            return np.zeros(len(X))
        # Чем ниже score (decision_function), тем более аномально; преобразуем к положительным значениям
        raw = self.model.decision_function(X)
        norm = (raw.max() - raw)  # инверсия
        if norm.max() > 0:
            norm = norm / norm.max()
        return norm


# --- Новая унифицированная абстракция + статистический baseline ---
class AnomalyModelProtocol:
    """Протокол единого интерфейса аномальной модели.

    Методы:
      update(x: Dict[str,float]) -> float  (возвращает score)
      is_anomaly(score: float) -> bool
    """
    def update(self, x: Dict[str, float]) -> float:  # pragma: no cover - интерфейс
        raise NotImplementedError
    def is_anomaly(self, score: float) -> bool:  # pragma: no cover
        raise NotImplementedError


class StreamingHalfSpaceTreesAdapter(AnomalyModelProtocol):
    def __init__(self, threshold: float = 0.7):
        if not RIVER_AVAILABLE:
            raise ImportError("river недоступен")
        self.model = anomaly.HalfSpaceTrees()
        self.threshold = threshold
    def update(self, x: Dict[str, float]) -> float:
        s = self.model.score_one(x)
        self.model.learn_one(x)
        return s
    def is_anomaly(self, score: float) -> bool:
        return score > self.threshold


class StatsQuantileBaseline(AnomalyModelProtocol):
    """Простой потоковый baseline: поддерживает скользящую медиану и MAD для одного агрегированного признака.

    Используем rms_a (или первую доступную rms_*). Score = |x - median| / (MAD+eps).
    Аномалия если score > z_threshold.
    """
    def __init__(self, z_threshold: float = 6.0, window: int = 5000):
        self.values: List[float] = []
        self.window = window
        self.z_threshold = z_threshold
    def update(self, x: Dict[str, float]) -> float:
        # Выбор признака (в порядке приоритета)
        for k in ('rms_a','rms_b','rms_c'):
            if k in x and x[k] is not None:
                v = float(x[k])
                break
        else:
            v = float(next(iter(x.values()), 0.0))
        self.values.append(v)
        if len(self.values) > self.window:
            self.values = self.values[-self.window:]
        import numpy as np
        arr = np.array(self.values)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med))) + 1e-9
        score = abs(v - med) / mad
        return score
    def is_anomaly(self, score: float) -> bool:
        return score > self.z_threshold

try:
    __all__  # type: ignore
except NameError:  # pragma: no cover
    __all__ = []  # type: ignore
__all__ += ['AnomalyModelProtocol','StreamingHalfSpaceTreesAdapter','StatsQuantileBaseline']

__all__ = [
    'HalfSpaceTreesIncremental', 'SlidingIsolationForest', 'IncrementalAnomalyResult',
    'ADWINDriftMonitor', 'PageHinkleyDriftMonitor', 'StreamDriftMonitor'
]


class ADWINDriftMonitor:
    """Обёртка для river.drift.ADWIN."""
    def __init__(self, delta: float = 0.002):
        if not RIVER_AVAILABLE:
            raise ImportError("river не установлен")
        self.detector = drift.ADWIN(delta=delta)

    def update(self, value: float) -> Dict[str, float | bool]:
        prior_width = self.detector.width
        in_drift, in_warning = self.detector.update(value)
        return {
            'value': value,
            'width': self.detector.width,
            'delta_width': self.detector.width - prior_width,
            'drift': bool(in_drift),
            'warning': bool(in_warning)
        }


class PageHinkleyDriftMonitor:
    """Обёртка для river.drift.PageHinkley."""
    def __init__(self, threshold: float = 50.0, min_instances: int = 30):
        if not RIVER_AVAILABLE:
            raise ImportError("river не установлен")
        self.detector = drift.PageHinkley(threshold=threshold, min_instances=min_instances)

    def update(self, value: float) -> Dict[str, float | bool]:
        in_drift, in_warning = self.detector.update(value)
        return {
            'value': value,
            'drift': bool(in_drift),
            'warning': bool(in_warning),
            'x_mean': getattr(self.detector, 'x_mean', None)
        }


class StreamDriftMonitor:
    """Комбинированный монитор, применяющий несколько детекторов дрейфа к поступающему значению.

    Использование:
        monitor = StreamDriftMonitor(use_adwin=True, use_ph=True)
        result = monitor.update(metric_value)
    """
    def __init__(self, use_adwin: bool = True, use_ph: bool = True):
        self.adwin = ADWINDriftMonitor() if use_adwin and RIVER_AVAILABLE else None
        self.page_hinkley = PageHinkleyDriftMonitor() if use_ph and RIVER_AVAILABLE else None

    def update(self, value: float) -> Dict[str, Dict]:
        out: Dict[str, Dict] = {}
        if self.adwin:
            out['adwin'] = self.adwin.update(value)
        if self.page_hinkley:
            out['page_hinkley'] = self.page_hinkley.update(value)
        return out
