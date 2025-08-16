"""Основные Celery задачи (перенос из tasks.py чтобы устранить конфликт пакета)."""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

import numpy as np
from celery import Task
from celery.signals import worker_ready, worker_shutdown
from sqlalchemy import select

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import (
	RawSignal, Feature, Equipment, Prediction,
	ProcessingStatus
)
from src.data_processing.feature_extraction import FeatureExtractor
from src.ml.train import load_latest_models
from src.ml.forecasting import RMSTrendForecaster
from src.utils.logger import get_logger, get_audit_logger
from src.worker.config import celery_app
from src.utils.serialization import load_float32_array
from src.utils.metrics import observe_latency as _observe_latency

settings = get_settings()
logger = get_logger(__name__)
audit_logger = get_audit_logger()

# --- Остальной код будет импортирован из оригинального файла tasks.py позже ---
