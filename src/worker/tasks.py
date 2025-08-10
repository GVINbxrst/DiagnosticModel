"""
Celery задачи для фоновой обработки данных диагностики двигателей

Этот модуль содержит асинхронные задачи для:
- Обработки сырых сигналов и извлечения признаков
- Применения моделей обнаружения аномалий
- Прогнозирования трендов RMS по фазам
- Автоматической переобработки при сбоях
"""

import asyncio
import gzip
import json
import traceback
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID

import numpy as np
from celery import Celery, Task
from celery.exceptions import Retry, MaxRetriesExceededError
from celery.signals import worker_ready, worker_shutdown
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import (
    RawSignal, Feature, Equipment, Prediction,
    ProcessingStatus, DefectType
)
from src.data_processing.feature_extraction import FeatureExtractor
from src.ml.train import load_latest_models
from src.ml.forecasting import RMSTrendForecaster
from src.utils.logger import get_logger
from src.worker.celery_app import celery_app
from src.worker.monitoring import track_task_metrics, get_worker_metrics_collector
from src.utils.metrics import (
    track_csv_processing, track_anomaly_detection,
    track_forecast_generation, increment_counter, observe_histogram
)

# Настройки
settings = get_settings()
logger = get_logger(__name__)
audit_logger = get_audit_logger()

# Инициализируем мониторинг Worker
worker_metrics = get_worker_metrics_collector()

# Конфигурация Celery
celery_app = Celery(
    'diagmod_worker',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=['src.worker.tasks']
)

# Настройки Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 час максимум на задачу
    task_soft_time_limit=3300,  # 55 минут мягкий лимит
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    task_compression='gzip',
    result_compression='gzip',
    task_routes={
        'src.worker.tasks.process_raw': {'queue': 'processing'},
        'src.worker.tasks.detect_anomalies': {'queue': 'ml'},
        'src.worker.tasks.forecast_trend': {'queue': 'ml'},
        'src.worker.tasks.cleanup_old_data': {'queue': 'maintenance'},
    },
    beat_schedule={
        'cleanup-old-data': {
            'task': 'src.worker.tasks.cleanup_old_data',
            'schedule': 3600.0,  # Каждый час
        },
        'retrain-models': {
            'task': 'src.worker.tasks.retrain_models',
            'schedule': 24 * 3600.0,  # Каждый день
        },
    }
)


class DatabaseTask(Task):
    """Базовый класс для задач с поддержкой БД"""

    def __init__(self):
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка ошибок задач"""
        self.logger.error(
            f"Задача {task_id} завершилась с ошибкой: {exc}",
            extra={
                'task_id': task_id,
                'args': args,
                'kwargs': kwargs,
                'traceback': str(einfo)
            }
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Обработка повторных попыток"""
        self.logger.warning(
            f"Повторная попытка задачи {task_id}: {exc}",
            extra={
                'task_id': task_id,
                'retry_count': self.request.retries,
                'args': args,
                'kwargs': kwargs
            }
        )

    def on_success(self, retval, task_id, args, kwargs):
        """Обработка успешного завершения"""
        self.logger.info(
            f"Задача {task_id} успешно завершена",
            extra={
                'task_id': task_id,
                'result': retval,
                'args': args,
                'kwargs': kwargs
            }
        )


async def decompress_signal_data(compressed_data: bytes) -> np.ndarray:
    """Распаковка сжатых сигнальных данных"""
    try:
        decompressed = gzip.decompress(compressed_data)
        # Восстанавливаем float32 массив
        return np.frombuffer(decompressed, dtype=np.float32)
    except Exception as e:
        logger.error(f"Ошибка распаковки данных: {e}")
        raise


async def compress_and_store_results(data: Any) -> bytes:
    """Сжатие результатов для хранения"""
    try:
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        return gzip.compress(json_str.encode('utf-8'))
    except Exception as e:
        logger.error(f"Ошибка сжатия результатов: {e}")
        raise


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    retry_backoff=True,
    retry_jitter=True
)
def process_raw(self, raw_id: str) -> Dict:
    """
    Обработка сырого сигнала: извлечение признаков

    Args:
        raw_id: UUID сырого сигнала

    Returns:
        Результат обработки с извлеченными признаками
    """
    task_start = datetime.utcnow()
    self.logger.info(f"Начинаем обработку сырого сигнала {raw_id}")

    try:
        # Запускаем асинхронную обработку
        result = asyncio.run(_process_raw_async(raw_id))

        processing_time = (datetime.utcnow() - task_start).total_seconds()
        result['processing_time_seconds'] = processing_time

        self.logger.info(
            f"Сигнал {raw_id} успешно обработан за {processing_time:.2f} сек",
            extra={'features_extracted': len(result.get('feature_ids', []))}
        )

        return result

    except Exception as exc:
        self.logger.error(f"Ошибка обработки сигнала {raw_id}: {exc}")

        # Обновляем статус в БД при ошибке
        asyncio.run(_update_signal_status(raw_id, ProcessingStatus.FAILED, str(exc)))

        # Повторяем попытку если не достигли лимита
        if self.request.retries < self.max_retries:
            self.logger.info(f"Повторная попытка {self.request.retries + 1}/{self.max_retries}")
            raise self.retry(countdown=60 * (self.request.retries + 1))

        raise exc


async def _process_raw_async(raw_id: str) -> Dict:
    """Асинхронная обработка сырого сигнала"""

    async with get_async_session() as session:
        # Загружаем сырой сигнал
        query = select(RawSignal).where(RawSignal.id == UUID(raw_id))
        result = await session.execute(query)
        raw_signal = result.scalar_one_or_none()

        if not raw_signal:
            raise ValueError(f"Сырой сигнал {raw_id} не найден")

        # Обновляем статус на "в обработке"
        await _update_signal_status(raw_id, ProcessingStatus.PROCESSING)

        try:
            # Распаковываем данные фаз
            phase_data = {}

            if raw_signal.phase_a:
                phase_data['phase_a'] = await decompress_signal_data(raw_signal.phase_a)

            if raw_signal.phase_b:
                phase_data['phase_b'] = await decompress_signal_data(raw_signal.phase_b)

            if raw_signal.phase_c:
                phase_data['phase_c'] = await decompress_signal_data(raw_signal.phase_c)

            if not phase_data:
                raise ValueError("Нет данных ни для одной фазы")

            # Извлекаем признаки
            extractor = FeatureExtractor(
                sample_rate=raw_signal.sample_rate or 25600
            )

            # Обрабатываем сигнал с временными окнами
            feature_ids = await extractor.process_raw_signal(
                raw_signal_id=UUID(raw_id),
                window_duration_ms=1000,  # Окна по 1 секунде
                overlap_ratio=0.5,        # 50% перекрытие
                session=session
            )

            # Обновляем статус на "завершено"
            await _update_signal_status(raw_id, ProcessingStatus.COMPLETED)

            # Автоматически запускаем обнаружение аномалий для всех извлеченных признаков
            for feature_id in feature_ids:
                detect_anomalies.delay(str(feature_id))

            return {
                'status': 'success',
                'raw_signal_id': raw_id,
                'feature_ids': [str(fid) for fid in feature_ids],
                'phases_processed': list(phase_data.keys()),
                'total_features': len(feature_ids)
            }

        except Exception as e:
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, str(e))
            raise


async def _update_signal_status(
    raw_id: str,
    status: ProcessingStatus,
    error_message: Optional[str] = None
):
    """Обновление статуса обработки сигнала"""
    async with get_async_session() as session:
        update_data = {
            'processing_status': status,
            'updated_at': datetime.utcnow()
        }

        if error_message:
            update_data['error_message'] = error_message

        query = update(RawSignal).where(
            RawSignal.id == UUID(raw_id)
        ).values(**update_data)

        await session.execute(query)
        await session.commit()


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 2, 'countdown': 30},
    retry_backoff=True
)
def detect_anomalies(self, feature_id: str) -> Dict:
    """
    Применение моделей обнаружения аномалий к извлеченным признакам

    Args:
        feature_id: UUID записи с признаками

    Returns:
        Результат детекции аномалий
    """
    task_start = datetime.utcnow()
    self.logger.info(f"Начинаем детекцию аномалий для признаков {feature_id}")

    try:
        result = asyncio.run(_detect_anomalies_async(feature_id))

        processing_time = (datetime.utcnow() - task_start).total_seconds()
        result['processing_time_seconds'] = processing_time

        self.logger.info(
            f"Детекция аномалий для {feature_id} завершена за {processing_time:.2f} сек",
            extra={
                'anomaly_detected': result.get('anomaly_detected', False),
                'confidence': result.get('confidence', 0)
            }
        )

        return result

    except Exception as exc:
        self.logger.error(f"Ошибка детекции аномалий {feature_id}: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(countdown=30 * (self.request.retries + 1))

        raise exc


async def _detect_anomalies_async(feature_id: str) -> Dict:
    """Асинхронная детекция аномалий"""

    async with get_async_session() as session:
        # Загружаем признаки
        query = select(Feature).where(Feature.id == UUID(feature_id))
        result = await session.execute(query)
        feature = result.scalar_one_or_none()

        if not feature:
            raise ValueError(f"Признаки {feature_id} не найдены")

        # Загружаем обученные модели
        models = await load_latest_models_async()

        if not models:
            raise RuntimeError("Обученные модели не найдены. Необходимо запустить обучение.")

        # Подготавливаем признаки для модели
        feature_vector = _prepare_feature_vector(feature)

        # Применяем модели
        isolation_result = None
        dbscan_result = None

        try:
            if 'isolation_forest' in models:
                isolation_pred = models['isolation_forest'].predict([feature_vector])[0]
                isolation_score = models['isolation_forest'].decision_function([feature_vector])[0]
                isolation_result = {
                    'prediction': int(isolation_pred),  # -1 = аномалия, 1 = норма
                    'score': float(isolation_score),
                    'is_anomaly': isolation_pred == -1
                }

            if 'dbscan' in models and 'preprocessor' in models:
                # Для DBSCAN нужны нормализованные данные
                normalized_features = models['preprocessor'].transform([feature_vector])
                dbscan_pred = models['dbscan'].predict(normalized_features)[0]
                dbscan_result = {
                    'cluster': int(dbscan_pred),
                    'is_anomaly': dbscan_pred == -1  # -1 = выброс
                }

        except Exception as e:
            logger.warning(f"Ошибка применения модели: {e}")

        # Определяем итоговый результат
        anomaly_detected = False
        confidence = 0.0

        if isolation_result and isolation_result['is_anomaly']:
            anomaly_detected = True
            confidence = max(confidence, abs(isolation_result['score']))

        if dbscan_result and dbscan_result['is_anomaly']:
            anomaly_detected = True
            confidence = max(confidence, 0.8)  # Высокая уверенность для DBSCAN

        # Нормализуем confidence в диапазон [0, 1]
        confidence = min(1.0, max(0.0, confidence))

        # Сохраняем результат в БД
        prediction = Prediction(
            feature_id=UUID(feature_id),
            model_name='ensemble_anomaly_detection',
            model_version='1.0.0',
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            prediction_data={
                'isolation_forest': isolation_result,
                'dbscan': dbscan_result,
                'feature_vector_size': len(feature_vector)
            },
            created_at=datetime.utcnow()
        )

        session.add(prediction)
        await session.commit()

        # Если аномалия обнаружена с высокой уверенностью, запускаем прогнозирование
        if anomaly_detected and confidence > 0.7:
            # Получаем equipment_id через связанные таблицы
            equipment_query = select(RawSignal.equipment_id).join(Feature).where(
                Feature.id == UUID(feature_id)
            )
            eq_result = await session.execute(equipment_query)
            equipment_id = eq_result.scalar_one_or_none()

            if equipment_id:
                forecast_trend.delay(str(equipment_id))

        return {
            'status': 'success',
            'feature_id': feature_id,
            'prediction_id': str(prediction.id),
            'anomaly_detected': anomaly_detected,
            'confidence': confidence,
            'models_applied': list(models.keys())
        }


def _prepare_feature_vector(feature: Feature) -> List[float]:
    """Подготовка вектора признаков для ML модели"""
    vector = []

    # RMS значения
    for phase in ['a', 'b', 'c']:
        rms_value = getattr(feature, f'rms_{phase}', None)
        vector.append(float(rms_value) if rms_value is not None else 0.0)

    # Crest factor
    for phase in ['a', 'b', 'c']:
        crest_value = getattr(feature, f'crest_{phase}', None)
        vector.append(float(crest_value) if crest_value is not None else 0.0)

    # Kurtosis
    for phase in ['a', 'b', 'c']:
        kurt_value = getattr(feature, f'kurtosis_{phase}', None)
        vector.append(float(kurt_value) if kurt_value is not None else 0.0)

    # Skewness
    for phase in ['a', 'b', 'c']:
        skew_value = getattr(feature, f'skewness_{phase}', None)
        vector.append(float(skew_value) if skew_value is not None else 0.0)

    # FFT пики (если есть)
    if feature.fft_spectrum:
        fft_data = feature.fft_spectrum
        if isinstance(fft_data, dict) and 'peaks' in fft_data:
            peaks = fft_data['peaks'][:5]  # Топ-5 пиков
            for peak in peaks:
                vector.append(float(peak.get('frequency', 0)))
                vector.append(float(peak.get('amplitude', 0)))

            # Дополняем до фиксированного размера
            while len(peaks) < 5:
                vector.extend([0.0, 0.0])
                peaks.append({})
    else:
        vector.extend([0.0] * 10)  # 5 пиков * 2 значения

    return vector


async def load_latest_models_async():
    """Асинхронная загрузка последних обученных моделей"""
    try:
        return await asyncio.get_event_loop().run_in_executor(
            None, load_latest_models
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        return {}


@celery_app.task(
    bind=True,
    base=DatabaseTask,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 2, 'countdown': 120},
    retry_backoff=True
)
def forecast_trend(self, equipment_id: str) -> Dict:
    """
    Прогнозирование трендов RMS для оборудования

    Args:
        equipment_id: UUID оборудования

    Returns:
        Результат прогнозирования
    """
    task_start = datetime.utcnow()
    self.logger.info(f"Начинаем прогнозирование трендов для оборудования {equipment_id}")

    try:
        result = asyncio.run(_forecast_trend_async(equipment_id))

        processing_time = (datetime.utcnow() - task_start).total_seconds()
        result['processing_time_seconds'] = processing_time

        self.logger.info(
            f"Прогнозирование для {equipment_id} завершено за {processing_time:.2f} сек",
            extra={
                'max_anomaly_probability': result.get('summary', {}).get('max_anomaly_probability', 0),
                'recommendation': result.get('summary', {}).get('recommendation', 'N/A')
            }
        )

        return result

    except Exception as exc:
        self.logger.error(f"Ошибка прогнозирования для {equipment_id}: {exc}")

        if self.request.retries < self.max_retries:
            raise self.retry(countdown=120 * (self.request.retries + 1))

        raise exc


async def _forecast_trend_async(equipment_id: str) -> Dict:
    """Асинхронное прогнозирование трендов"""

    async with get_async_session() as session:
        # Проверяем существование оборудования
        query = select(Equipment).where(Equipment.id == UUID(equipment_id))
        result = await session.execute(query)
        equipment = result.scalar_one_or_none()

        if not equipment:
            raise ValueError(f"Оборудование {equipment_id} не найдено")

        # Создаем прогнозировщик
        forecaster = RMSTrendForecaster()

        # Выполняем прогнозирование для всех фаз
        forecast_result = await forecaster.forecast_equipment_trends(
            equipment_id=UUID(equipment_id),
            session=session,
            phases=['a', 'b', 'c'],
            forecast_steps=24  # Прогноз на 24 часа
        )

        # Сохраняем результат в сжатом виде для больших данных
        compressed_result = await compress_and_store_results(forecast_result)

        # Сохраняем краткую сводку в таблицу прогнозов
        summary = forecast_result.get('summary', {})

        prediction = Prediction(
            equipment_id=UUID(equipment_id),
            model_name='rms_trend_forecasting',
            model_version='1.0.0',
            anomaly_detected=summary.get('max_anomaly_probability', 0) > 0.5,
            confidence=summary.get('max_anomaly_probability', 0),
            prediction_data={
                'forecast_summary': summary,
                'phases_analyzed': list(forecast_result.get('phases', {}).keys()),
                'forecast_steps': forecast_result.get('forecast_steps', 24),
                'compressed_full_result_size': len(compressed_result)
            },
            created_at=datetime.utcnow()
        )

        session.add(prediction)
        await session.commit()

        return {
            'status': 'success',
            'equipment_id': equipment_id,
            'prediction_id': str(prediction.id),
            'summary': summary,
            'forecast_steps': forecast_result.get('forecast_steps', 24),
            'phases_analyzed': list(forecast_result.get('phases', {}).keys())
        }


# Дополнительные служебные задачи

@celery_app.task(bind=True, base=DatabaseTask)
def cleanup_old_data(self, days_to_keep: int = 30) -> Dict:
    """
    Очистка старых данных и результатов

    Args:
        days_to_keep: Количество дней для хранения данных
    """
    self.logger.info(f"Начинаем очистку данных старше {days_to_keep} дней")

    try:
        result = asyncio.run(_cleanup_old_data_async(days_to_keep))
        self.logger.info(f"Очистка завершена: удалено {result['total_deleted']} записей")
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка очистки данных: {exc}")
        raise exc


async def _cleanup_old_data_async(days_to_keep: int) -> Dict:
    """Асинхронная очистка старых данных"""

    cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
    deleted_counts = {}

    async with get_async_session() as session:
        # Удаляем старые прогнозы
        old_predictions = await session.execute(
            select(Prediction).where(Prediction.created_at < cutoff_date)
        )
        predictions_to_delete = old_predictions.scalars().all()

        for prediction in predictions_to_delete:
            await session.delete(prediction)

        deleted_counts['predictions'] = len(predictions_to_delete)

        # Удаляем старые необработанные сигналы
        old_signals = await session.execute(
            select(RawSignal).where(
                RawSignal.created_at < cutoff_date,
                RawSignal.processing_status == ProcessingStatus.FAILED
            )
        )
        signals_to_delete = old_signals.scalars().all()

        for signal in signals_to_delete:
            await session.delete(signal)

        deleted_counts['failed_signals'] = len(signals_to_delete)

        await session.commit()

    total_deleted = sum(deleted_counts.values())

    return {
        'status': 'success',
        'cutoff_date': cutoff_date.isoformat(),
        'deleted_counts': deleted_counts,
        'total_deleted': total_deleted
    }


@celery_app.task(bind=True, base=DatabaseTask)
def retrain_models(self) -> Dict:
    """Переобучение моделей на новых данных"""

    self.logger.info("Начинаем переобучение моделей")

    try:
        # Импортируем и запускаем обучение
        from src.ml.train import AnomalyModelTrainer

        result = asyncio.run(_retrain_models_async())

        self.logger.info("Переобучение моделей завершено успешно")
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка переобучения моделей: {exc}")
        raise exc


async def _retrain_models_async() -> Dict:
    """Асинхронное переобучение моделей"""

    trainer = AnomalyModelTrainer()

    # Обучаем на всех доступных данных
    training_results = await trainer.train_models(
        equipment_ids=None,  # Все оборудование
        contamination=0.1,   # 10% ожидаемых аномалий
        save_visualizations=False  # Не сохраняем визуализации в автоматическом режиме
    )

    return {
        'status': 'success',
        'training_results': training_results,
        'retrain_timestamp': datetime.utcnow().isoformat()
    }


# Обработчики событий worker

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Обработчик готовности worker"""
    logger.info("Celery worker готов к работе")


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Обработчик остановки worker"""
    logger.info("Celery worker завершает работу")


# Экспорт для использования в других модулях
__all__ = [
    'celery_app',
    'process_raw',
    'detect_anomalies',
    'forecast_trend',
    'cleanup_old_data',
    'retrain_models'
]
