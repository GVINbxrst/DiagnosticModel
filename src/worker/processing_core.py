"""Общая (единая) реализация обработки RawSignal.

Содержит повторяющуюся ранее логику из tasks.py и tasks_logic.py:
 - Переходы статусов (PENDING -> PROCESSING -> COMPLETED / FAILED)
 - Подгрузка CSV при отсутствии фазовых данных
 - Валидация данных
 - Извлечение признаков

Использование:
  from .processing_core import process_raw_core
  await process_raw_core(raw_id)
"""
from __future__ import annotations

from typing import Dict, List, Sequence
from uuid import UUID

import numpy as np
from sqlalchemy import select

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import RawSignal, ProcessingStatus, Forecast, AnomalyScore, Feature, StreamStat
from src.utils.serialization import load_float32_array
from src.utils.logger import get_logger

# Ленивая загрузка тяжёлых зависимостей внутри функций
logger = get_logger(__name__)


async def _update_signal_status(raw_id: str | UUID, status: ProcessingStatus, error: str | None = None):
    rid = str(raw_id)
    async with get_async_session() as session:
        res = await session.execute(select(RawSignal).where(RawSignal.id == UUID(rid)))
        raw = res.scalar_one_or_none()
        if not raw:
            return
        raw.processing_status = status
        if status == ProcessingStatus.COMPLETED:
            raw.processed = True  # type: ignore[attr-defined]
        if error:
            raw.meta = (raw.meta or {}) | {"error": error}
        await session.commit()


async def _decompress_signal_data(compressed: bytes):  # -> np.ndarray | None
    try:
        return load_float32_array(compressed)
    except Exception:  # pragma: no cover
        return None


async def process_raw_core(raw_id: str) -> Dict:
    """Основной конвейер обработки RawSignal.

    Глобальный fail-safe: любая неперехваченная ошибка -> статус FAILED.
    Частичные ошибки (валидация, отсутствие фаз, feature extraction) уже устанавливают FAILED локально.
    """
    settings = get_settings()
    try:
        async with get_async_session() as session:
            res = await session.execute(select(RawSignal).where(RawSignal.id == UUID(raw_id)))
            raw_signal = res.scalar_one_or_none()
            if not raw_signal:
                raise ValueError("RawSignal не найден")
            if raw_signal.processing_status == ProcessingStatus.PROCESSING:
                return {"status": "in_progress", "raw_signal_id": raw_id}
            if raw_signal.processing_status == ProcessingStatus.COMPLETED:
                return {"status": "skipped", "raw_signal_id": raw_id}
            await _update_signal_status(raw_id, ProcessingStatus.PROCESSING)

        # 1. Подгрузка CSV при отсутствии данных
        loaded_from_csv = False
        if not any([raw_signal.phase_a, raw_signal.phase_b, raw_signal.phase_c]) and raw_signal.file_name:
            from src.data_processing.csv_loader import CSVLoader, CSVLoaderError, InvalidCSVFormatError  # type: ignore
            csv_path = settings.data_path / raw_signal.file_name
            loader = CSVLoader()
            try:
                if csv_path.exists():
                    await loader.load_file(csv_path, raw_id=raw_signal.id, sample_rate=raw_signal.sample_rate_hz)
                    loaded_from_csv = True
                    await session.refresh(raw_signal)
                else:
                    raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
            except (CSVLoaderError, InvalidCSVFormatError, FileNotFoundError) as e:
                await _update_signal_status(raw_id, ProcessingStatus.FAILED, f"csv_loader: {e}")
                raise

        # 2. Распаковка фаз
        phase_arrays: Dict[str, np.ndarray] = {}
        for attr, key in [("phase_a", "R"), ("phase_b", "S"), ("phase_c", "T")]:
            comp = getattr(raw_signal, attr, None)
            if comp:
                arr = await _decompress_signal_data(comp)
                if arr is not None:
                    phase_arrays[key] = arr
        if not phase_arrays:
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, "Нет данных фаз")
            raise ValueError("Нет данных фаз для обработки")

        # 3. Валидация
        from src.data_processing.data_validator import DataValidator  # type: ignore
        validator = DataValidator()
        phase_lists = {k: v.tolist() for k, v in phase_arrays.items()}
        validation_results = validator.validate_csv_data(phase_lists, raw_signal.sample_rate_hz, filename=raw_signal.file_name or raw_id)
        validation_summary = validator.get_validation_summary(validation_results)
        if validation_summary.get("has_critical_errors"):
            raw_signal.meta = (raw_signal.meta or {}) | {"validation": validation_summary}
            await session.commit()
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, "validation_failed")
            return {"status": "validation_failed", "raw_signal_id": raw_id, "validation": validation_summary}

        # 4. Извлечение признаков
        from src.data_processing.feature_extraction import FeatureExtractor, InsufficientDataError  # type: ignore
        extractor = FeatureExtractor(sample_rate=raw_signal.sample_rate_hz or 25600)
        try:
            feature_ids: Sequence[UUID] = await extractor.process_raw_signal(UUID(raw_id), window_duration_ms=1000, overlap_ratio=0.5)
        except InsufficientDataError as exc:
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, f"feature_extraction: {exc}")
            raise

        # 4a+. Эмбеддинги автоэнкодера (best-effort)
        try:
            from src.ml.feature_extraction import EmbeddingExtractor
            # Загружаем один «репрезентативный» сигнал (фаза A или первая доступная)
            rep_phase = None
            for attr in ["phase_a", "phase_b", "phase_c"]:
                comp = getattr(raw_signal, attr, None)
                if comp:
                    rep_phase = await _decompress_signal_data(comp)
                    if rep_phase is not None:
                        break
            if rep_phase is not None and len(feature_ids) > 0:
                emb_extractor = EmbeddingExtractor(raw_signal.sample_rate_hz or 25600)
                # Минимальное (быстрое) обучение по нескольким сегментам: берём первые 8 окон если есть
                train_segments = []
                seg_len = min(4096, len(rep_phase))
                if seg_len > 64:
                    step = max(1, seg_len // 8)
                    for i in range(0, min(len(rep_phase)-seg_len, step*8), step):
                        train_segments.append(rep_phase[i:i+seg_len])
                if train_segments:
                    try:
                        emb_extractor.train_autoencoder(train_segments, epochs=1, batch_size=4)
                    except Exception as te:  # pragma: no cover
                        logger.debug(f"embed train skip: {te}")
                # Генерируем embedding
                try:
                    emb_res = emb_extractor.embed_signal(rep_phase)
                    # Записываем в extra.embedding для всех новых features (или только первой — здесь всем)
                    feat_res = await session.execute(select(Feature).where(Feature.id.in_(feature_ids)))
                    feat_rows = feat_res.scalars().all()
                    for fr in feat_rows:
                        extra = fr.extra or {}
                        extra['embedding'] = emb_res.embedding
                        fr.extra = extra
                    await session.commit()
                except Exception as ee:  # pragma: no cover
                    logger.debug(f"embed inference skip: {ee}")
        except Exception as e_emb:  # pragma: no cover
            logger.debug(f"embedding pipeline skipped: {e_emb}")

        # 4b. Поточное вычисление аномальных скорингов (stream / baseline) – best effort
        try:
            from pathlib import Path
            models_root = settings.models_path / 'anomaly_detection' / 'latest'
            manifest_path = models_root / 'manifest.json'
            if manifest_path.exists():
                import json, joblib
                data = json.loads(manifest_path.read_text(encoding='utf-8'))
                mtype = data.get('model_type')
                threshold = float(data.get('threshold', 0.7))
                if mtype == 'stream':
                    stream_state = models_root / 'stream_state.pkl'
                    if stream_state.exists():
                        state = joblib.load(stream_state)
                        from src.ml.incremental import StreamingHalfSpaceTreesAdapter
                        model = StreamingHalfSpaceTreesAdapter(threshold=threshold)
                        try:
                            model.model = state.get('model')  # type: ignore
                        except Exception:
                            pass
                        if feature_ids:
                            feat_res = await session.execute(select(Feature).where(Feature.id.in_(feature_ids)))
                            feat_rows = feat_res.scalars().all()
                            for fr in feat_rows:
                                rec = {
                                    'rms_a': float(fr.rms_a) if fr.rms_a is not None else None,
                                    'rms_b': float(fr.rms_b) if fr.rms_b is not None else None,
                                    'rms_c': float(fr.rms_c) if fr.rms_c is not None else None,
                                }
                                vec = {k: v for k, v in rec.items() if v is not None}
                                try:
                                    score = model.update(vec)
                                    is_anomaly = model.is_anomaly(score)
                                    session.add(AnomalyScore(
                                        feature_id=fr.id,
                                        raw_id=raw_signal.id,
                                        equipment_id=raw_signal.equipment_id,
                                        model_type='stream',
                                        model_version='latest',
                                        score=float(score),
                                        is_anomaly=bool(is_anomaly),
                                        threshold=threshold,
                                        meta={'online': True}
                                    ))
                                except Exception as se:  # pragma: no cover
                                    logger.debug(f"stream anomaly score error: {se}")
                            await session.commit()
                elif mtype == 'stats':
                    # baseline stats не реализуем для онлайн скоринга здесь (score уже считается при detect)
                    pass
        except Exception as inc_e:  # pragma: no cover
            logger.debug(f"incremental scoring skipped: {inc_e}")

        # 4c. Мониторинг дрейфа распределения (ADWIN / PageHinkley) по простой метрике (rms_a) если есть river
        try:
            from src.ml.incremental import StreamDriftMonitor, RIVER_AVAILABLE
            if RIVER_AVAILABLE and feature_ids:
                # Берём первую фичу для мониторинга как пример (можно расширить)
                feat_res = await session.execute(select(Feature).where(Feature.id.in_(feature_ids)))
                feat_rows = feat_res.scalars().all()
                monitor = StreamDriftMonitor()
                for fr in feat_rows:
                    val = fr.rms_a
                    if val is None:
                        continue
                    drift_results = monitor.update(float(val))
                    for det_name, det_res in drift_results.items():
                        session.add(StreamStat(
                            equipment_id=raw_signal.equipment_id,
                            raw_id=raw_signal.id,
                            feature_id=fr.id,
                            detector=det_name,
                            metric='rms_a',
                            value=float(det_res.get('value', 0.0)),
                            drift_detected=bool(det_res.get('drift', False)),
                            details={k: v for k, v in det_res.items() if k not in {'value', 'drift'}}
                        ))
                await session.commit()
        except Exception as drift_e:  # pragma: no cover
            logger.debug(f"drift monitor skipped: {drift_e}")

        raw_signal.meta = (raw_signal.meta or {}) | {
            "validation": validation_summary,
            "feature_count": len(feature_ids),
            "loaded_from_csv": loaded_from_csv,
        }
        await session.commit()

        # 5. Прогноз (best-effort, не ломает основную обработку)
        forecast_record_id = None
        try:
            from src.ml.forecasting import forecast_rms, InsufficientDataError as ForecastInsufficient
            fc_res = await forecast_rms(raw_signal.equipment_id)
            forecast_record = Forecast(
                raw_id=raw_signal.id,
                equipment_id=raw_signal.equipment_id,
                horizon=len(fc_res.get('forecast', [])),
                method=fc_res.get('model', 'unknown'),
                forecast_data=fc_res,
                probability_over_threshold=fc_res.get('probability_over_threshold'),
                model_version='v1'
            )
            session.add(forecast_record)
            await session.commit()
            forecast_record_id = str(forecast_record.id)
        except ForecastInsufficient:
            logger.info("Прогноз пропущен: недостаточно данных")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Ошибка прогноза: {e}")

        # 6. Финализация статуса
        await _update_signal_status(raw_id, ProcessingStatus.COMPLETED)
        return {
            "status": "success",
            "raw_signal_id": raw_id,
            "feature_ids": [str(f) for f in feature_ids],
            "validation": validation_summary,
            "forecast_id": forecast_record_id
        }
    except Exception as e:
        # Глобальный catch: устанавливаем FAILED если ещё не установлен
        try:
            await _update_signal_status(raw_id, ProcessingStatus.FAILED, str(e)[:200])
        except Exception:  # pragma: no cover
            pass
        logger.exception(f"process_raw_core failed: {e}")
        return {"status": "failed", "raw_signal_id": raw_id, "error": str(e)}
