"""
Роутер для получения сигналов по фазам для визуализации
"""

import gzip
import time
from typing import List, Optional
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import get_current_user, require_any_role
from src.api.middleware.metrics import metrics
from src.api.schemas import SignalResponse, PhaseData, UserInfo, SignalListResponse, SignalListItem
from src.database.connection import get_async_session
from src.database.models import RawSignal, Equipment, ProcessingStatus
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get("/signals/{raw_id}", response_model=SignalResponse)
async def get_signal_data(
    raw_id: UUID,
    downsample_factor: Optional[int] = Query(None, ge=1, le=100, description="Фактор прореживания данных"),
    start_sample: Optional[int] = Query(None, ge=0, description="Начальный отсчет"),
    end_sample: Optional[int] = Query(None, ge=0, description="Конечный отсчет"),
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Получение данных сигнала по фазам для визуализации

    Возвращает:
    - Сырые данные сигналов по всем доступным фазам
    - Метаданные сигнала (частота дискретизации, время записи)
    - Базовую статистику по каждой фазе
    - Возможность прореживания данных для быстрой визуализации
    """

    start_time = time.time()

    # Получаем сигнал из БД
    signal_query = select(RawSignal).where(RawSignal.id == raw_id)
    signal_result = await session.execute(signal_query)
    raw_signal = signal_result.scalar_one_or_none()

    if not raw_signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Сигнал с ID {raw_id} не найден"
        )

    try:
        # Распаковываем данные фаз
        phases = []

        for phase_name, phase_data in [('a', raw_signal.phase_a), ('b', raw_signal.phase_b), ('c', raw_signal.phase_c)]:
            if phase_data is not None:
                # Распаковываем сжатые данные
                decompressed_data = gzip.decompress(phase_data)
                values = np.frombuffer(decompressed_data, dtype=np.float32)

                # Применяем фильтрацию по диапазону
                if start_sample is not None or end_sample is not None:
                    start_idx = start_sample or 0
                    end_idx = end_sample or len(values)
                    values = values[start_idx:end_idx]

                # Применяем прореживание если указано
                if downsample_factor and downsample_factor > 1:
                    values = values[::downsample_factor]

                # Рассчитываем статистику
                statistics = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'rms': float(np.sqrt(np.mean(values**2))),
                    'peak_to_peak': float(np.ptp(values))
                }

                phase_info = PhaseData(
                    phase_name=phase_name.upper(),
                    values=values.tolist(),
                    has_data=True,
                    samples_count=len(values),
                    statistics=statistics
                )
            else:
                # Фаза отсутствует
                phase_info = PhaseData(
                    phase_name=phase_name.upper(),
                    values=[],
                    has_data=False,
                    samples_count=0,
                    statistics=None
                )

            phases.append(phase_info)

        # Подготавливаем метаданные
        metadata = raw_signal.metadata or {}
        metadata.update({
            'downsample_factor': downsample_factor,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'file_hash': raw_signal.file_hash
        })

        processing_time = time.time() - start_time

        # Обновляем метрики
        metrics.observe_histogram(
            'api_request_duration_seconds',
            processing_time,
            {'method': 'GET', 'endpoint': '/signals/{id}'}
        )

        logger.info(
            f"Сигнал {raw_id} отдан пользователю {current_user.username} за {processing_time:.3f}s",
            extra={
                'signal_id': str(raw_id),
                'equipment_id': str(raw_signal.equipment_id),
                'phases_count': sum(1 for p in phases if p.has_data),
                'downsample_factor': downsample_factor
            }
        )

        return SignalResponse(
            raw_signal_id=raw_signal.id,
            equipment_id=raw_signal.equipment_id,
            recorded_at=raw_signal.recorded_at,
            sample_rate=raw_signal.sample_rate,
            total_samples=raw_signal.samples_count,
            phases=phases,
            metadata=metadata,
            processing_status=raw_signal.processing_status
        )

    except Exception as e:
        logger.error(f"Ошибка обработки сигнала {raw_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка обработки данных сигнала"
        )


@router.get("/signals", response_model=SignalListResponse)
async def list_signals(
    equipment_id: Optional[UUID] = Query(None, description="Фильтр по оборудованию"),
    status_filter: Optional[List[ProcessingStatus]] = Query(None, description="Фильтр по статусу обработки"),
    page: int = Query(1, ge=1, description="Номер страницы"),
    page_size: int = Query(20, ge=1, le=100, description="Размер страницы"),
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение списка сигналов с фильтрацией и пагинацией"""

    # Базовый запрос
    signals_query = select(RawSignal)

    # Применяем фильтры
    if equipment_id:
        signals_query = signals_query.where(RawSignal.equipment_id == equipment_id)

    if status_filter:
        signals_query = signals_query.where(RawSignal.processing_status.in_(status_filter))

    # Подсчет общего количества
    from sqlalchemy import func
    count_query = select(func.count(RawSignal.id))
    if equipment_id:
        count_query = count_query.where(RawSignal.equipment_id == equipment_id)
    if status_filter:
        count_query = count_query.where(RawSignal.processing_status.in_(status_filter))

    total_count_result = await session.execute(count_query)
    total_count = total_count_result.scalar()

    # Пагинация и сортировка
    offset = (page - 1) * page_size
    signals_query = signals_query.order_by(RawSignal.recorded_at.desc()).offset(offset).limit(page_size)

    # Выполняем запрос
    signals_result = await session.execute(signals_query)
    raw_signals = signals_result.scalars().all()

    # Преобразуем в схемы
    signal_items = []
    for signal in raw_signals:
        # Определяем доступные фазы
        phases_available = []
        if signal.phase_a is not None:
            phases_available.append('A')
        if signal.phase_b is not None:
            phases_available.append('B')
        if signal.phase_c is not None:
            phases_available.append('C')

        # Извлекаем имя файла из метаданных
        file_name = None
        if signal.metadata:
            file_name = signal.metadata.get('original_filename')

        item = SignalListItem(
            raw_signal_id=signal.id,
            equipment_id=signal.equipment_id,
            recorded_at=signal.recorded_at,
            samples_count=signal.samples_count,
            phases_available=phases_available,
            processing_status=signal.processing_status,
            file_name=file_name
        )
        signal_items.append(item)

    has_next = total_count > page * page_size

    return SignalListResponse(
        signals=signal_items,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=has_next
    )


@router.get("/signals/{raw_id}/preview")
async def get_signal_preview(
    raw_id: UUID,
    samples: int = Query(1000, ge=100, le=10000, description="Количество отсчетов для превью"),
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Получение превью сигнала для быстрой визуализации

    Возвращает прореженные данные для предварительного просмотра
    """

    # Получаем сигнал
    signal_query = select(RawSignal).where(RawSignal.id == raw_id)
    signal_result = await session.execute(signal_query)
    raw_signal = signal_result.scalar_one_or_none()

    if not raw_signal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Сигнал с ID {raw_id} не найден"
        )

    try:
        preview_data = {
            'signal_id': str(raw_id),
            'equipment_id': str(raw_signal.equipment_id),
            'sample_rate': raw_signal.sample_rate,
            'total_samples': raw_signal.samples_count,
            'preview_samples': samples,
            'phases': {}
        }

        # Рассчитываем фактор прореживания
        downsample_factor = max(1, raw_signal.samples_count // samples)

        for phase_name, phase_data in [('A', raw_signal.phase_a), ('B', raw_signal.phase_b), ('C', raw_signal.phase_c)]:
            if phase_data is not None:
                # Распаковываем и прореживаем
                decompressed_data = gzip.decompress(phase_data)
                values = np.frombuffer(decompressed_data, dtype=np.float32)

                # Прореживание
                preview_values = values[::downsample_factor][:samples]

                preview_data['phases'][phase_name] = {
                    'values': preview_values.tolist(),
                    'samples_count': len(preview_values),
                    'downsample_factor': downsample_factor,
                    'statistics': {
                        'mean': float(np.mean(preview_values)),
                        'std': float(np.std(preview_values)),
                        'min': float(np.min(preview_values)),
                        'max': float(np.max(preview_values))
                    }
                }
            else:
                preview_data['phases'][phase_name] = {
                    'values': [],
                    'samples_count': 0,
                    'has_data': False
                }

        return preview_data

    except Exception as e:
        logger.error(f"Ошибка генерации превью для сигнала {raw_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка генерации превью сигнала"
        )


@router.get("/equipment", response_model=List[dict])
async def list_equipment(
    current_user: UserInfo = Depends(require_any_role),
    session: AsyncSession = Depends(get_async_session)
):
    """Получение списка доступного оборудования"""

    equipment_query = select(Equipment).where(Equipment.is_active == True).order_by(Equipment.name)
    equipment_result = await session.execute(equipment_query)
    equipment_list = equipment_result.scalars().all()

    return [
        {
            'id': str(eq.id),
            'name': eq.name,
            'equipment_type': eq.equipment_type,
            'model': eq.model,
            'location': eq.location,
            'installed_at': eq.installed_at.isoformat() if eq.installed_at else None,
            'last_maintenance': eq.last_maintenance.isoformat() if eq.last_maintenance else None
        }
        for eq in equipment_list
    ]
