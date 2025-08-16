"""Адаптационный пакет `src.worker.tasks`.

Из-за конфликта имён (файл `tasks.py` и пакет `tasks/`) Python выбирает пакет,
а не файл. Нам нужно прозрачно предоставить объекты из файла `tasks.py`.

Решение: динамически загружаем файл `tasks.py` под внутренним именем
`src.worker._tasks_file` через importlib.spec_from_file_location и
проксируем атрибуты. Это минимально инвазивно и не требует переименований.
"""

from __future__ import annotations

import pathlib
import importlib.util
import sys
from types import ModuleType
from typing import Optional

_EXPORTED = [
	'process_raw', 'detect_anomalies', 'forecast_trend',
	'cleanup_old_data', 'retrain_models',
	'_process_raw_async', '_detect_anomalies_async', '_forecast_trend_async',
	'decompress_signal_data', 'compress_and_store_results',
	'_prepare_feature_vector', '_update_signal_status', 'get_async_session',
	# Дополнительно для unit тестов (patch targets)
	'FeatureExtractor', 'load_latest_models_async', 'RMSTrendForecaster'
]

_TASKS_FILE_MODULE: Optional[ModuleType] = None

def _load_tasks_file() -> ModuleType:  # pragma: no cover - инфраструктура
	global _TASKS_FILE_MODULE
	if _TASKS_FILE_MODULE is not None:
		return _TASKS_FILE_MODULE
	base_dir = pathlib.Path(__file__).resolve().parent.parent
	tasks_path = base_dir / 'tasks.py'
	spec = importlib.util.spec_from_file_location('src.worker._tasks_file', tasks_path)
	if spec is None or spec.loader is None:
		raise ImportError('Cannot load tasks.py module spec')
	mod = importlib.util.module_from_spec(spec)
	sys.modules.setdefault('src.worker._tasks_file', mod)
	spec.loader.exec_module(mod)  # type: ignore
	_TASKS_FILE_MODULE = mod
	return mod

def __getattr__(name: str):  # pragma: no cover - инфраструктура
	if name in _EXPORTED:
		if name == 'get_async_session':
			# Берём напрямую из database.connection чтобы patch('src.worker.tasks.get_async_session') работал
			from src.database.connection import get_async_session  # type: ignore
			return get_async_session
		mod = _load_tasks_file()
		# Приводим имя celery задач к ожидаемому пространству имён tests (src.worker.tasks.*)
		if name in {'process_raw','detect_anomalies','forecast_trend'}:
			task_obj = getattr(mod, name)
			try:
				task_obj.name = f'src.worker.tasks.{name}'  # обновляем зарегистрированное имя
			except Exception:
				pass
		return getattr(mod, name)
	raise AttributeError(name)

def __dir__():  # pragma: no cover
	return sorted(list(globals().keys()) + _EXPORTED)

__all__ = list(_EXPORTED)
