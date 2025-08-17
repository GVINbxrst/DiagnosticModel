from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from src.database.connection import db_session
from src.database.models import RawSignal, ProcessingStatus, Feature
from src.api.middleware.auth import require_any_role, UserInfo
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get('/pipeline/status')
async def get_pipeline_status(
    equipment_id: Optional[str] = None,
    session: AsyncSession = Depends(db_session),
    current_user: UserInfo = Depends(require_any_role)
):
    """Возвращает агрегированный статус пайплайна загрузки/обработки.

    Метрики:
      total_raw: всего сырых сигналов
      pending: ожидают обработки
      processing: в обработке
      completed: успешно обработаны (processed=true)
      failed: завершились с ошибкой
      features_total: всего рассчитанных окон признаков
      last_recorded_at: метка времени последнего сигнала
    Optional: фильтр по equipment_id (UUID в строковом виде)
    """
    try:
        filters = []
        if equipment_id:
            from uuid import UUID
            try:
                eq_uuid = UUID(equipment_id)
                filters.append(RawSignal.equipment_id == eq_uuid)
            except Exception:
                raise HTTPException(status_code=400, detail='invalid_equipment_id')

        # Базовый запрос подсчётов по статусам
        base_q = select(
            func.count(RawSignal.id),
            func.sum(case((RawSignal.processing_status == ProcessingStatus.PENDING, 1), else_=0)).label('pending'),
            func.sum(case((RawSignal.processing_status == ProcessingStatus.PROCESSING, 1), else_=0)).label('processing'),
            func.sum(case((RawSignal.processing_status == ProcessingStatus.COMPLETED, 1), else_=0)).label('completed'),
            func.sum(case((RawSignal.processing_status == ProcessingStatus.FAILED, 1), else_=0)).label('failed'),
            func.max(RawSignal.recorded_at).label('last_recorded_at')
        )
        if filters:
            for f in filters:
                base_q = base_q.where(f)
        res = await session.execute(base_q)
        row = res.first()
        total_raw = row[0] if row else 0
        pending = row[1] if row else 0
        processing = row[2] if row else 0
        completed = row[3] if row else 0
        failed = row[4] if row else 0
        last_recorded_at = row[5].isoformat() if row and row[5] else None

        # Количество признаков (features)
        feat_q = select(func.count(Feature.id))
        if filters:
            # join через raw_id
            from sqlalchemy import join
            from sqlalchemy.orm import aliased
            # простой вариант: подзапрос raw ids
            sub_raw = select(RawSignal.id).where(*filters).subquery()
            feat_q = feat_q.where(Feature.raw_id.in_(select(sub_raw.c.id)))
        feat_res = await session.execute(feat_q)
        features_total = feat_res.scalar() or 0

        return {
            'total_raw': int(total_raw or 0),
            'pending': int(pending or 0),
            'processing': int(processing or 0),
            'completed': int(completed or 0),
            'failed': int(failed or 0),
            'features_total': int(features_total or 0),
            'last_recorded_at': last_recorded_at,
            'equipment_id': equipment_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'pipeline status error: {e}')
        raise HTTPException(status_code=500, detail='pipeline_status_internal_error')
