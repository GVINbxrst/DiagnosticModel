# Адаптационный пакет: проксирует объекты из tasks.py при конфликте имён

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
	# Patch targets
	'FeatureExtractor', 'load_latest_models_async', 'RMSTrendForecaster',
	'clustering_pipeline', 'run_clustering_async'
]

_TASKS_FILE_MODULE: Optional[ModuleType] = None

def _load_tasks_file() -> ModuleType:  # pragma: no cover
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

def __getattr__(name: str):  # pragma: no cover
	if name in _EXPORTED:
		if name == 'get_async_session':
			# Возврат прямой функции сессии (для patch в тестах)
			from src.database.connection import get_async_session  # type: ignore
			return get_async_session
		mod = _load_tasks_file()
	# Обновляем имя celery-задачи под пространство src.worker.tasks.*
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
