"""
Специализированные задачи для обработки данных диагностики

Этот модуль содержит дополнительные задачи для:
- Пакетной обработки файлов
- Мониторинга состояния системы
- Автоматической диагностики оборудования
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID

from celery import group, chain, chord
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import RawSignal, Feature, Equipment, Prediction, ProcessingStatus
from src.data_processing.csv_loader import CSVLoader
from src.utils.logger import get_logger
from src.worker.tasks import celery_app, DatabaseTask, process_raw, detect_anomalies, forecast_trend

settings = get_settings()
logger = get_logger(__name__)


@celery_app.task(bind=True, base=DatabaseTask)
def batch_process_directory(self, directory_path: str, equipment_id: Optional[str] = None) -> Dict:
    """
    Пакетная обработка всех CSV файлов в директории

    Args:
        directory_path: Путь к директории с CSV файлами
        equipment_id: Опциональный ID оборудования

    Returns:
        Результат пакетной обработки
    """
    self.logger.info(f"Начинаем пакетную обработку директории: {directory_path}")

    try:
        result = asyncio.run(_batch_process_directory_async(directory_path, equipment_id))
        self.logger.info(f"Пакетная обработка завершена: {result['total_files']} файлов")
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка пакетной обработки: {exc}")
        raise exc


async def _batch_process_directory_async(directory_path: str, equipment_id: Optional[str]) -> Dict:
    """Асинхронная пакетная обработка директории"""

    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Директория не найдена: {directory_path}")

    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"CSV файлы не найдены в директории: {directory_path}")

    loader = CSVLoader()
    processed_files = []
    failed_files = []

    # Определяем equipment_id если не указан
    if not equipment_id:
        equipment_id = await _get_or_create_equipment_from_filename(csv_files[0].name)

    for csv_file in csv_files:
        try:
            # Загружаем CSV файл
            stats = await loader.load_csv_file(
                file_path=str(csv_file),
                equipment_id=UUID(equipment_id)
            )

            processed_files.append({
                'file': str(csv_file),
                'stats': stats.__dict__,
                'status': 'success'
            })

            # Запускаем обработку каждого загруженного сигнала
            if hasattr(stats, 'raw_signal_ids'):
                for raw_id in stats.raw_signal_ids:
                    process_raw.delay(str(raw_id))

        except Exception as e:
            logger.error(f"Ошибка загрузки файла {csv_file}: {e}")
            failed_files.append({
                'file': str(csv_file),
                'error': str(e),
                'status': 'failed'
            })

    return {
        'status': 'success',
        'directory': directory_path,
        'equipment_id': equipment_id,
        'total_files': len(csv_files),
        'processed_files': len(processed_files),
        'failed_files': len(failed_files),
        'details': {
            'processed': processed_files,
            'failed': failed_files
        }
    }


async def _get_or_create_equipment_from_filename(filename: str) -> str:
    """Получение или создание оборудования на основе имени файла"""

    # Простая логика извлечения ID из имени файла
    # Например: motor_001.csv -> EQ_2025_000001
    base_name = Path(filename).stem
    equipment_name = base_name.replace('_', ' ').title()

    async with get_async_session() as session:
        # Ищем существующее оборудование
        query = select(Equipment).where(Equipment.name.ilike(f"%{equipment_name}%"))
        result = await session.execute(query)
        equipment = result.scalar_one_or_none()

        if equipment:
            return str(equipment.id)

        # Создаем новое оборудование
        new_equipment = Equipment(
            name=equipment_name,
            equipment_type='motor',
            model=f"Auto-detected from {filename}",
            location='Unknown',
            is_active=True,
            created_at=datetime.utcnow()
        )

        session.add(new_equipment)
        await session.commit()
        await session.refresh(new_equipment)

        logger.info(f"Создано новое оборудование: {equipment_name} ({new_equipment.id})")

        return str(new_equipment.id)


@celery_app.task(bind=True, base=DatabaseTask)
def process_equipment_workflow(self, equipment_id: str, force_reprocess: bool = False) -> Dict:
    """
    Полный рабочий процесс обработки для конкретного оборудования

    Args:
        equipment_id: ID оборудования
        force_reprocess: Принудительная переобработка уже обработанных данных

    Returns:
        Результат полного цикла обработки
    """
    self.logger.info(f"Начинаем полный цикл обработки для оборудования {equipment_id}")

    try:
        result = asyncio.run(_process_equipment_workflow_async(equipment_id, force_reprocess))
        self.logger.info(f"Полный цикл для {equipment_id} завершен")
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка полного цикла для {equipment_id}: {exc}")
        raise exc


async def _process_equipment_workflow_async(equipment_id: str, force_reprocess: bool) -> Dict:
    """Асинхронный полный цикл обработки оборудования"""

    async with get_async_session() as session:
        # Получаем все сырые сигналы для оборудования
        status_filter = [ProcessingStatus.PENDING]
        if force_reprocess:
            status_filter.extend([ProcessingStatus.COMPLETED, ProcessingStatus.FAILED])

        query = select(RawSignal).where(
            RawSignal.equipment_id == UUID(equipment_id),
            RawSignal.processing_status.in_(status_filter)
        )

        result = await session.execute(query)
        raw_signals = result.scalars().all()

        if not raw_signals:
            return {
                'status': 'no_data',
                'equipment_id': equipment_id,
                'message': 'Нет сигналов для обработки'
            }

        # Создаем цепочку задач для каждого сигнала
        workflow_tasks = []

        for raw_signal in raw_signals:
            # Цепочка: обработка -> детекция аномалий
            signal_workflow = chain(
                process_raw.s(str(raw_signal.id)),
                # После успешной обработки запускаем детекцию для всех извлеченных признаков
                _create_anomaly_detection_chord.s()
            )
            workflow_tasks.append(signal_workflow)

        # Запускаем все цепочки параллельно
        job = group(workflow_tasks)
        workflow_result = job.apply_async()

        # После завершения всех обработок запускаем прогнозирование
        forecast_task = forecast_trend.apply_async(
            args=[equipment_id],
            countdown=300  # Запускаем через 5 минут после начала обработки
        )

        return {
            'status': 'started',
            'equipment_id': equipment_id,
            'signals_to_process': len(raw_signals),
            'workflow_job_id': workflow_result.id,
            'forecast_task_id': forecast_task.id,
            'force_reprocess': force_reprocess
        }


@celery_app.task
def _create_anomaly_detection_chord(process_result: Dict) -> str:
    """Создание chord задачи для детекции аномалий"""

    if process_result.get('status') != 'success':
        return f"Обработка сигнала не удалась: {process_result}"

    feature_ids = process_result.get('feature_ids', [])
    if not feature_ids:
        return "Нет извлеченных признаков для детекции"

    # Создаем задачи детекции для всех признаков
    detection_tasks = [detect_anomalies.s(feature_id) for feature_id in feature_ids]

    # Запускаем все детекции параллельно
    detection_job = group(detection_tasks)
    result = detection_job.apply_async()

    return f"Запущена детекция аномалий для {len(feature_ids)} признаков: {result.id}"


@celery_app.task(bind=True, base=DatabaseTask)
def health_check_system(self) -> Dict:
    """
    Проверка состояния системы и мониторинг

    Returns:
        Состояние системы и статистики
    """
    self.logger.info("Выполняем проверку состояния системы")

    try:
        result = asyncio.run(_health_check_system_async())
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка проверки состояния системы: {exc}")
        raise exc


async def _health_check_system_async() -> Dict:
    """Асинхронная проверка состояния системы"""

    async with get_async_session() as session:
        # Статистика по сырым сигналам
        raw_stats = await session.execute(
            select(
                RawSignal.processing_status,
                func.count(RawSignal.id).label('count')
            ).group_by(RawSignal.processing_status)
        )

        raw_signal_stats = {row.processing_status.value: row.count for row in raw_stats}

        # Статистика по признакам
        features_count = await session.execute(
            select(func.count(Feature.id))
        )
        total_features = features_count.scalar()

        # Статистика по прогнозам за последние 24 часа
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_predictions = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.created_at >= yesterday
            )
        )
        recent_predictions_count = recent_predictions.scalar()

        # Статистика по аномалиям за последние 24 часа
        recent_anomalies = await session.execute(
            select(func.count(Prediction.id)).where(
                Prediction.created_at >= yesterday,
                Prediction.anomaly_detected == True
            )
        )
        recent_anomalies_count = recent_anomalies.scalar()

        # Активное оборудование
        active_equipment = await session.execute(
            select(func.count(Equipment.id)).where(
                Equipment.is_active == True
            )
        )
        active_equipment_count = active_equipment.scalar()

        # Оборудование с недавними аномалиями
        equipment_with_anomalies = await session.execute(
            select(func.count(func.distinct(Prediction.equipment_id))).where(
                Prediction.created_at >= yesterday,
                Prediction.anomaly_detected == True
            )
        )
        equipment_with_anomalies_count = equipment_with_anomalies.scalar()

        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': {
                'raw_signals': {
                    'by_status': raw_signal_stats,
                    'total': sum(raw_signal_stats.values())
                },
                'features': {
                    'total': total_features
                },
                'predictions_24h': {
                    'total': recent_predictions_count,
                    'anomalies': recent_anomalies_count,
                    'anomaly_rate': recent_anomalies_count / max(1, recent_predictions_count)
                },
                'equipment': {
                    'active': active_equipment_count,
                    'with_recent_anomalies': equipment_with_anomalies_count,
                    'anomaly_equipment_rate': equipment_with_anomalies_count / max(1, active_equipment_count)
                }
            },
            'alerts': _generate_system_alerts(
                raw_signal_stats,
                recent_anomalies_count,
                equipment_with_anomalies_count,
                active_equipment_count
            )
        }


def _generate_system_alerts(
    raw_signal_stats: Dict,
    recent_anomalies: int,
    equipment_with_anomalies: int,
    active_equipment: int
) -> List[Dict]:
    """Генерация системных алертов"""

    alerts = []

    # Проверка количества неудачных обработок
    failed_signals = raw_signal_stats.get('failed', 0)
    total_signals = sum(raw_signal_stats.values())

    if total_signals > 0 and failed_signals / total_signals > 0.1:  # >10% неудач
        alerts.append({
            'level': 'warning',
            'type': 'processing_failures',
            'message': f'Высокий процент неудачных обработок: {failed_signals}/{total_signals} ({failed_signals/total_signals*100:.1f}%)'
        })

    # Проверка количества аномалий
    if recent_anomalies > 50:  # Много аномалий за день
        alerts.append({
            'level': 'warning',
            'type': 'high_anomaly_count',
            'message': f'Обнаружено много аномалий за последние 24 часа: {recent_anomalies}'
        })

    # Проверка доли оборудования с аномалиями
    if active_equipment > 0 and equipment_with_anomalies / active_equipment > 0.3:  # >30% оборудования
        alerts.append({
            'level': 'critical',
            'type': 'widespread_anomalies',
            'message': f'Аномалии обнаружены на большом количестве оборудования: {equipment_with_anomalies}/{active_equipment} ({equipment_with_anomalies/active_equipment*100:.1f}%)'
        })

    # Проверка застрявших задач
    pending_signals = raw_signal_stats.get('processing', 0)
    if pending_signals > 100:
        alerts.append({
            'level': 'warning',
            'type': 'processing_backlog',
            'message': f'Большое количество сигналов в очереди обработки: {pending_signals}'
        })

    return alerts


@celery_app.task(bind=True, base=DatabaseTask)
def daily_equipment_report(self, equipment_id: Optional[str] = None) -> Dict:
    """
    Ежедневный отчет по состоянию оборудования

    Args:
        equipment_id: ID конкретного оборудования или None для всего

    Returns:
        Ежедневный отчет
    """
    self.logger.info(f"Генерируем ежедневный отчет для оборудования: {equipment_id or 'всё'}")

    try:
        result = asyncio.run(_daily_equipment_report_async(equipment_id))
        return result

    except Exception as exc:
        self.logger.error(f"Ошибка генерации отчета: {exc}")
        raise exc


async def _daily_equipment_report_async(equipment_id: Optional[str]) -> Dict:
    """Асинхронная генерация ежедневного отчета"""

    async with get_async_session() as session:
        yesterday = datetime.utcnow() - timedelta(days=1)

        # Фильтр по оборудованию
        equipment_filter = []
        if equipment_id:
            equipment_filter.append(Equipment.id == UUID(equipment_id))

        # Статистика по оборудованию
        equipment_query = select(Equipment).where(*equipment_filter, Equipment.is_active == True)
        equipment_result = await session.execute(equipment_query)
        equipment_list = equipment_result.scalars().all()

        equipment_reports = []

        for equipment in equipment_list:
            # Аномалии за день
            anomalies_query = select(func.count(Prediction.id)).where(
                Prediction.equipment_id == equipment.id,
                Prediction.created_at >= yesterday,
                Prediction.anomaly_detected == True
            )
            anomalies_count = (await session.execute(anomalies_query)).scalar()

            # Последний прогноз
            last_forecast_query = select(Prediction).where(
                Prediction.equipment_id == equipment.id,
                Prediction.model_name == 'rms_trend_forecasting'
            ).order_by(Prediction.created_at.desc()).limit(1)

            last_forecast_result = await session.execute(last_forecast_query)
            last_forecast = last_forecast_result.scalar_one_or_none()

            # Статус оборудования
            status = 'normal'
            if anomalies_count > 10:
                status = 'critical'
            elif anomalies_count > 5:
                status = 'warning'
            elif anomalies_count > 0:
                status = 'attention'

            equipment_reports.append({
                'equipment_id': str(equipment.id),
                'equipment_name': equipment.name,
                'status': status,
                'anomalies_24h': anomalies_count,
                'last_forecast': {
                    'timestamp': last_forecast.created_at.isoformat() if last_forecast else None,
                    'max_anomaly_probability': last_forecast.confidence if last_forecast else None,
                    'recommendation': last_forecast.prediction_data.get('forecast_summary', {}).get('recommendation') if last_forecast and last_forecast.prediction_data else None
                }
            })

        # Общая сводка
        total_equipment = len(equipment_reports)
        critical_equipment = len([r for r in equipment_reports if r['status'] == 'critical'])
        warning_equipment = len([r for r in equipment_reports if r['status'] == 'warning'])
        total_anomalies = sum(r['anomalies_24h'] for r in equipment_reports)

        return {
            'report_date': yesterday.date().isoformat(),
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_equipment': total_equipment,
                'critical_equipment': critical_equipment,
                'warning_equipment': warning_equipment,
                'normal_equipment': total_equipment - critical_equipment - warning_equipment,
                'total_anomalies_24h': total_anomalies
            },
            'equipment_details': equipment_reports,
            'recommendations': _generate_daily_recommendations(equipment_reports)
        }


def _generate_daily_recommendations(equipment_reports: List[Dict]) -> List[str]:
    """Генерация рекомендаций на основе ежедневного отчета"""

    recommendations = []

    critical_equipment = [r for r in equipment_reports if r['status'] == 'critical']
    warning_equipment = [r for r in equipment_reports if r['status'] == 'warning']

    if critical_equipment:
        recommendations.append(
            f"🔴 КРИТИЧНО: Немедленно проверьте оборудование: {', '.join([e['equipment_name'] for e in critical_equipment])}"
        )

    if warning_equipment:
        recommendations.append(
            f"🟡 ВНИМАНИЕ: Запланируйте проверку оборудования: {', '.join([e['equipment_name'] for e in warning_equipment])}"
        )

    high_anomaly_equipment = [r for r in equipment_reports if r['anomalies_24h'] > 20]
    if high_anomaly_equipment:
        recommendations.append(
            f"📊 Повышенная активность аномалий: {', '.join([e['equipment_name'] for e in high_anomaly_equipment])}"
        )

    if not critical_equipment and not warning_equipment:
        recommendations.append("✅ Все оборудование работает в нормальном режиме")

    return recommendations


# Экспорт дополнительных задач
__all__ = [
    'batch_process_directory',
    'process_equipment_workflow',
    'health_check_system',
    'daily_equipment_report'
]
