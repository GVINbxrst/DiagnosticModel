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

from src.api.middleware.auth import get_current_user, require_operator
from src.api.middleware.metrics import track_file_upload, metrics
from src.api.schemas import UploadResponse, UploadMetadata, UserInfo
from src.data_processing.csv_loader import CSVLoader
from src.database.connection import get_async_session
from src.database.models import Equipment
from src.worker.tasks import process_raw
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/upload", response_model=UploadResponse)
async def upload_csv_file(
    file: UploadFile = File(..., description="CSV файл с токовыми с��гналами"),
    equipment_id: Optional[str] = Form(None, description="ID оборудования (UUID)"),
    sample_rate: Optional[int] = Form(25600, description="Частота дискретизации (Гц)"),
    description: Optional[str] = Form(None, description="Описание файла"),
    current_user: UserInfo = Depends(require_operator),
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

    # Обработка equipment_id
    target_equipment_id = None
    if equipment_id:
        try:
            target_equipment_id = UUID(equipment_id)

            # Проверяем существование оборудования
            from sqlalchemy import select
            query = select(Equipment).where(Equipment.id == target_equipment_id)
            result = await session.execute(query)
            equipment = result.scalar_one_or_none()

            if not equipment:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Оборудование с ID {equipment_id} не найдено"
                )

        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Неверный формат equipment_id (ожидается UUID)"
            )
    else:
        # Создаем новое оборудование на основе имени файла
        target_equipment_id = await _create_equipment_from_filename(
            file.filename, session, current_user.id
        )

    try:
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            # Сохраняем загруженный файл
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

            if not file_size:
                file_size = len(content)

        # Инициализируем CSV загрузчик
        csv_loader = CSVLoader()

        # Загружаем файл в базу данных
        stats = await csv_loader.load_csv_file(
            file_path=temp_file_path,
            equipment_id=target_equipment_id,
            sample_rate=sample_rate or 25600,
            metadata={
                'original_filename': file.filename,
                'uploaded_by': str(current_user.id),
                'upload_timestamp': time.time(),
                'description': description,
                'file_size_bytes': file_size
            }
        )

        # Удаляем временный файл
        os.unlink(temp_file_path)

        # Запускаем задачу обработки
        task = process_raw.delay(str(stats.raw_signal_id))

        # Определяем какие фазы были обнаружены
        phases_detected = []
        if stats.phase_a_samples > 0:
            phases_detected.append('A')
        if stats.phase_b_samples > 0:
            phases_detected.append('B')
        if stats.phase_c_samples > 0:
            phases_detected.append('C')

        # Обновляем метрики
        track_file_upload('csv', 'success', file_size)

        processing_time = time.time() - start_time

        logger.info(
            f"Файл {file.filename} успешно загружен пользователем {current_user.username}",
            extra={
                'equipment_id': str(target_equipment_id),
                'file_size': file_size,
                'samples_count': stats.total_samples,
                'phases_detected': phases_detected,
                'processing_time': processing_time
            }
        )

        return UploadResponse(
            success=True,
            message="Файл успешно загружен и поставлен в очередь обработки",
            raw_signal_id=stats.raw_signal_id,
            equipment_id=target_equipment_id,
            filename=file.filename,
            samples_count=stats.total_samples,
            phases_detected=phases_detected,
            processing_task_id=task.id,
            file_size_bytes=file_size,
            upload_time=stats.upload_time
        )

    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass

        # Обновляем метрики ошибок
        track_file_upload('csv', 'error', file_size)
        metrics.increment_counter('api_errors_total', {'status_code': '500', 'error_type': 'upload'})

        logger.error(f"Ошибка загрузки файла {file.filename}: {e}", exc_info=True)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка обработки файла: {str(e)}"
        )


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
async def get_upload_status(
    task_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
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
