"""
Роутер для загрузки CSV файлов с токовыми сигналами
"""

import os
import tempfile
import time
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.auth import require_operator
from src.utils.metrics import observe_latency
from src.api.schemas import UploadResponse, UserInfo
from src.data_processing.csv_loader import CSVLoader, InvalidCSVFormatError
from src.database.connection import get_async_session
from src.database.models import Equipment
from src.worker.tasks import process_raw
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload")
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/upload'})
async def upload_csv_file(
    file: UploadFile = File(..., description="CSV файл с токовыми сигналами"),
    equipment_id: Optional[str] = Form(None, description="ID оборудования (UUID)"),
    sample_rate: Optional[int] = Form(25600, description="Частота дискретизации (Гц)"),
    description: Optional[str] = Form(None, description="Описание файла"),
    current_user: Optional[UserInfo] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Загрузка CSV файла с токовыми сигналами

    Принимает CSV файл в формате:
    - Первая строка: заголовок вида `current_R,current_S,current_T`
    - Далее: строки значений трех фаз через запятую
    - Возможны пустые значения для фаз S и T

    После загрузки автоматически запускается задача извлечения признаков.
    """

    start_time = time.time()

    # Валидация файла
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Поддерживаются только CSV файлы"
        )

    # Проверка размера файла (максимум 500MB)
    file_size = 0
    if hasattr(file.file, 'seek') and hasattr(file.file, 'tell'):
        current_pos = file.file.tell()
        file.file.seek(0, 2)  # Конец файла
        file_size = file.file.tell()
        file.file.seek(current_pos)  # Возвращаемся к началу
        if file_size > 500 * 1024 * 1024:  # 500MB
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Размер файла превышает 500MB"
            )

    # Проверка equipment_id (если есть)
    target_equipment_id = None
    if equipment_id:
        try:
            target_equipment_id = UUID(equipment_id)
            from sqlalchemy import select
            query = select(Equipment).where(Equipment.id == target_equipment_id)
            eq = await session.execute(query)
            if not eq.scalar_one_or_none():
                raise HTTPException(status_code=422, detail="Оборудование не найдено")
        except Exception:
            raise HTTPException(status_code=422, detail="Некорректный equipment_id")

    # Сохраняем файл во временный файл
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        logger.error(f"Ошибка сохранения файла: {e}")
        raise HTTPException(status_code=500, detail="Ошибка сохранения файла")

    # Проверяем заголовок CSV
    try:
        with open(tmp_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip()
        if header.replace(' ', '') != 'current_R,current_S,current_T':
            os.unlink(tmp_path)
            raise HTTPException(status_code=422, detail="Некорректный заголовок CSV. Ожидается: current_R,current_S,current_T")
    except Exception as e:
        logger.error(f"Ошибка чтения заголовка: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail="Ошибка чтения файла")

    # Загружаем CSV через CSVLoader
    try:
        loader = CSVLoader()
        # Передаем расширенные метаданные
        stats_or_id = await loader.load_csv_file(
            tmp_path,
            equipment_id=target_equipment_id,
            sample_rate=sample_rate or 25600,
            metadata={
                'original_filename': file.filename,
                'uploaded_by': str(current_user.id) if current_user else None,
                'description': description,
                'file_size_bytes': file_size
            }
        )
    except InvalidCSVFormatError as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV: {e}")
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail="Ошибка обработки файла")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Ставим задачу Celery
    try:
        # CSVLoader сейчас возвращает CSVProcessingStats; обеспечим поддержку обоих вариантов
        raw_id = getattr(stats_or_id, 'raw_signal_id', None) or getattr(stats_or_id, 'raw_signal_ids', [None])[0] or stats_or_id
        task = process_raw.delay(str(raw_id))
    except Exception as e:
        logger.error(f"Ошибка постановки задачи Celery: {e}")
        raise HTTPException(status_code=500, detail="Ошибка постановки задачи Celery")

    return {"raw_id": str(raw_id), "task_id": str(task.id), "status": "queued"}


async def _create_equipment_from_filename(
    filename: str,
    session: AsyncSession,
    user_id: UUID
) -> UUID:
    """Создание нового оборудования на основе имени файла"""

    # Извлекаем имя без расширения
    base_name = os.path.splitext(filename)[0]
    equipment_name = base_name.replace('_', ' ').replace('-', ' ').title()

    # Создаем новое оборудование
    new_equipment = Equipment(
        id=uuid4(),
        name=equipment_name,
        equipment_type='motor',
        model=f"Auto-created from {filename}",
        location='Unknown',
        is_active=True,
        created_by=user_id,
        metadata={
            'auto_created': True,
            'source_filename': filename,
            'created_by_upload': True
        }
    )

    session.add(new_equipment)
    await session.commit()
    await session.refresh(new_equipment)

    logger.info(f"Создано новое оборудование: {equipment_name} ({new_equipment.id})")

    return new_equipment.id


@router.get("/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    """Получение статуса задачи загрузки"""

    from celery.result import AsyncResult
    from src.worker.config import celery_app

    try:
        result = AsyncResult(task_id, app=celery_app)

        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "progress": None,  # Можно добавить прогресс если нужно
            "error": str(result.traceback) if result.failed() else None
        }

    except Exception as e:
        logger.error(f"Ошибка получения статуса задачи {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка получения статуса задачи"
        )
