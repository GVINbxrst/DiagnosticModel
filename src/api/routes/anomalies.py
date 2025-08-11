"""
Роутер для получения аномалий по оборудованию
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.middleware.auth import get_current_user, require_any_role
from src.api.middleware.metrics import metrics
from src.api.schemas import (
    AnomaliesResponse, AnomalyInfo, ForecastInfo,
    UserInfo, AnomalyFilter, TimeRangeFilter, PaginationParams
)
from src.database.connection import get_async_session
from src.database.models import Equipment, Prediction, Feature, RawSignal
from src.utils.logger import get_logger
from src.utils.metrics import observe_latency

router = APIRouter()
logger = get_logger(__name__)


@router.get("/anomalies/{equipment_id}", response_model=AnomaliesResponse)
@observe_latency('api_request_duration_seconds', labels={'method':'GET','endpoint':'/anomalies/{id}'})
async def get_equipment_anomalies(
    equipment_id: UUID,
    start_date: Optional[datetime] = Query(None, description="Начальная дата фильтра"),
    end_date: Optional[datetime] = Query(None, description="Конечная дата фильтра"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Минимальная уверенность"),
    severity_levels: Optional[List[str]] = Query(None, description="Уровни критичности"),
    phases: Optional[List[str]] = Query(None, description="Фазы для фильтрации"),
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Получение списка аномалий для конкретного оборудования

    Возвращает:
    - Список обнаруженных аномалий с детальной информацией
    - Последний прогноз для оборудования
    - Сводную статистику по аномалиям
    """

    start_time = time.time()

    # Проверяем существование оборудования
    equipment_query = select(Equipment).where(Equipment.id == equipment_id)
    equipment_result = await session.execute(equipment_query)
    equipment = equipment_result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Оборудование с ID {equipment_id} не найдено"
        )

    # Устанавливаем временной диапазон по умолчанию (последние 30 дней)
    if not end_date:
        end_date = datetime.utcnow()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    # Базовый запрос аномалий
    anomalies_query = select(Prediction).join(Feature).join(RawSignal).where(
        and_(
            RawSignal.equipment_id == equipment_id,
            Prediction.anomaly_detected == True,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        )
    ).options(
        selectinload(Prediction.feature)
    )

    # Применяем фильтры
    if min_confidence is not None:
        anomalies_query = anomalies_query.where(Prediction.confidence >= min_confidence)

    # Подсчет общего количества
    count_query = select(func.count(Prediction.id)).select_from(
        Prediction.__table__.join(Feature.__table__).join(RawSignal.__table__)
    ).where(
        and_(
            RawSignal.equipment_id == equipment_id,
            Prediction.anomaly_detected == True,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        )
    )

    if min_confidence is not None:
        count_query = count_query.where(Prediction.confidence >= min_confidence)

    total_anomalies_result = await session.execute(count_query)
    total_anomalies = total_anomalies_result.scalar()

    # Пагинация
    offset = (page - 1) * page_size
    anomalies_query = anomalies_query.order_by(desc(Prediction.created_at)).offset(offset).limit(page_size)

    # Выполняем запрос аномалий
    anomalies_result = await session.execute(anomalies_query)
    predictions = anomalies_result.scalars().all()

    # Преобразуем в схемы
    anomalies = []
    for prediction in predictions:
        # Определяем затронутые фазы
        affected_phases = []
        feature = prediction.feature
        if feature:
            if feature.rms_a is not None:
                affected_phases.append('A')
            if feature.rms_b is not None:
                affected_phases.append('B')
            if feature.rms_c is not None:
                affected_phases.append('C')

        # Определяем критичность на основе уверенности
        if prediction.confidence >= 0.8:
            severity = "critical"
        elif prediction.confidence >= 0.6:
            severity = "high"
        elif prediction.confidence >= 0.4:
            severity = "medium"
        else:
            severity = "low"

        anomaly = AnomalyInfo(
            id=prediction.id,
            feature_id=prediction.feature_id,
            anomaly_type="statistical_deviation",  # Можно расширить
            confidence=prediction.confidence,
            severity=severity,
            description=f"Обнаружена аномалия с уверенностью {prediction.confidence:.2%}",
            detected_at=prediction.created_at,
            window_start=feature.window_start if feature else prediction.created_at,
            window_end=feature.window_end if feature else prediction.created_at,
            affected_phases=affected_phases,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            prediction_data=prediction.prediction_data or {}
        )
        anomalies.append(anomaly)

    # Получаем последний прогноз
    forecast_info = await _get_latest_forecast(equipment_id, session)

    # Формируем сводную статистику
    summary_stats = await _calculate_anomaly_summary(
        equipment_id, start_date, end_date, session
    )

    processing_time = time.time() - start_time

    # Метрика времени запроса теперь собирается декоратором observe_latency

    logger.info(
        f"Запрос аномалий для оборудования {equipment_id} выполнен за {processing_time:.3f}s",
        extra={
            'equipment_id': str(equipment_id),
            'anomalies_count': len(anomalies),
            'total_anomalies': total_anomalies,
            'user': current_user.username
        }
    )

    return AnomaliesResponse(
        equipment_id=equipment_id,
        anomalies=anomalies,
        forecast=forecast_info,
        total_anomalies=total_anomalies,
        period_start=start_date,
        period_end=end_date,
        summary=summary_stats
    )


async def _get_latest_forecast(equipment_id: UUID, session: AsyncSession) -> Optional[ForecastInfo]:
    """Получение последнего прогноза для оборудования"""

    # Ищем последний прогноз
    forecast_query = select(Prediction).where(
        and_(
            Prediction.equipment_id == equipment_id,
            Prediction.model_name == 'rms_trend_forecasting'
        )
    ).order_by(desc(Prediction.created_at)).limit(1)

    forecast_result = await session.execute(forecast_query)
    forecast_prediction = forecast_result.scalar_one_or_none()

    if not forecast_prediction:
        return None

    # Извлекаем данные прогноза
    prediction_data = forecast_prediction.prediction_data or {}
    forecast_summary = prediction_data.get('forecast_summary', {})

    return ForecastInfo(
        equipment_id=equipment_id,
        forecast_horizon_hours=24,  # По умолчанию 24 часа
        max_anomaly_probability=forecast_summary.get('max_anomaly_probability', 0.0),
        recommendation=forecast_summary.get('recommendation', 'Рекомендация не доступна'),
        generated_at=forecast_prediction.created_at,
        phases_analyzed=prediction_data.get('phases_analyzed', []),
        forecast_details=forecast_summary
    )


async def _calculate_anomaly_summary(
    equipment_id: UUID,
    start_date: datetime,
    end_date: datetime,
    session: AsyncSession
) -> dict:
    """Расчет сводной статистики по аномалиям"""

    # Статистика по критичности
    severity_query = select(
        func.count(Prediction.id).label('count'),
        func.avg(Prediction.confidence).label('avg_confidence'),
        func.max(Prediction.confidence).label('max_confidence')
    ).select_from(
        Prediction.__table__.join(Feature.__table__).join(RawSignal.__table__)
    ).where(
        and_(
            RawSignal.equipment_id == equipment_id,
            Prediction.anomaly_detected == True,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        )
    )

    severity_result = await session.execute(severity_query)
    severity_stats = severity_result.first()

    # Статистика по дням
    daily_query = select(
        func.date(Prediction.created_at).label('date'),
        func.count(Prediction.id).label('count')
    ).select_from(
        Prediction.__table__.join(Feature.__table__).join(RawSignal.__table__)
    ).where(
        and_(
            RawSignal.equipment_id == equipment_id,
            Prediction.anomaly_detected == True,
            Prediction.created_at >= start_date,
            Prediction.created_at <= end_date
        )
    ).group_by(func.date(Prediction.created_at))

    daily_result = await session.execute(daily_query)
    daily_stats = daily_result.fetchall()

    return {
        'total_anomalies': severity_stats.count if severity_stats else 0,
        'average_confidence': float(severity_stats.avg_confidence) if severity_stats and severity_stats.avg_confidence else 0.0,
        'max_confidence': float(severity_stats.max_confidence) if severity_stats and severity_stats.max_confidence else 0.0,
        'daily_distribution': [
            {'date': str(row.date), 'count': row.count}
            for row in daily_stats
        ] if daily_stats else [],
        'period_days': (end_date - start_date).days,
        'anomalies_per_day': (severity_stats.count / max(1, (end_date - start_date).days)) if severity_stats else 0.0
    }


@router.get("/anomalies/{equipment_id}/forecast")
@observe_latency('api_request_duration_seconds', labels={'method':'GET','endpoint':'/anomalies/{id}/forecast'})
async def get_equipment_forecast(
    equipment_id: UUID,
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение детального прогноза для оборудования"""

    # Проверяем существование оборудования
    equipment_query = select(Equipment).where(Equipment.id == equipment_id)
    equipment_result = await session.execute(equipment_query)
    equipment = equipment_result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Оборудование с ID {equipment_id} не найдено"
        )

    # Получаем детальный прогноз
    forecast_info = await _get_latest_forecast(equipment_id, session)

    if not forecast_info:
        # Если прогноза нет, запускаем его генерацию
        from src.worker.tasks import forecast_trend
        task = forecast_trend.delay(str(equipment_id))

        return {
            "message": "Прогноз генерируется",
            "task_id": task.id,
            "equipment_id": str(equipment_id),
            "estimated_completion_time": "2-5 минут"
        }

    return forecast_info


@router.post("/anomalies/{equipment_id}/reanalyze")
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/anomalies/{id}/reanalyze'})
async def reanalyze_equipment(
    equipment_id: UUID,
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """Запуск повторного анализа аномалий для оборудования"""

    # Проверяем существование оборудования
    equipment_query = select(Equipment).where(Equipment.id == equipment_id)
    equipment_result = await session.execute(equipment_query)
    equipment = equipment_result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Оборудование с ID {equipment_id} не найдено"
        )

    # Запускаем полный workflow анализа
    from src.worker.tasks.specialized_tasks import process_equipment_workflow
    task = process_equipment_workflow.delay(str(equipment_id), force_reprocess=True)

    logger.info(
        f"Запущен повторный анализ для оборудования {equipment_id} пользователем {current_user.username}"
    )

    return {
        "message": "Повторный анализ запущен",
        "task_id": task.id,
        "equipment_id": str(equipment_id),
        "estimated_completion_time": "10-30 минут"
    }
