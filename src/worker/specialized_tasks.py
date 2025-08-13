"""Специализированные задачи (перенос для устранения циклических импортов)."""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID
from celery import group, chain
from sqlalchemy import select, func
from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import RawSignal, Feature, Equipment, Prediction, ProcessingStatus
from src.data_processing.csv_loader import CSVLoader
from src.utils.logger import get_logger
from src.worker.config import celery_app

def _core_tasks():
    from src.worker.tasks import process_raw, detect_anomalies, forecast_trend  # noqa
    return process_raw, detect_anomalies, forecast_trend

settings = get_settings()
logger = get_logger(__name__)

@celery_app.task(bind=True)
def batch_process_directory(self, directory_path: str, equipment_id: Optional[str] = None) -> Dict:
    process_raw, *_ = _core_tasks()
    self.logger.info(f"Начинаем пакетную обработку директории: {directory_path}")
    try:
        result = asyncio.run(_batch_process_directory_async(directory_path, equipment_id, process_raw))
        return result
    except Exception as exc:
        self.logger.error(f"Ошибка пакетной обработки: {exc}")
        raise

async def _batch_process_directory_async(directory_path: str, equipment_id: Optional[str], process_raw_task) -> Dict:
    directory = Path(directory_path)
    if not directory.exists():
        raise ValueError(f"Директория не найдена: {directory_path}")
    csv_files = list(directory.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"CSV файлы не найдены в директории: {directory_path}")
    loader = CSVLoader()
    processed_files, failed_files = [], []
    if not equipment_id:
        equipment_id = await _get_or_create_equipment_from_filename(csv_files[0].name)
    for csv_file in csv_files:
        try:
            stats = await loader.load_csv_file(file_path=str(csv_file), equipment_id=UUID(equipment_id))
            processed_files.append({'file': str(csv_file), 'stats': stats.__dict__, 'status': 'success'})
            if hasattr(stats, 'raw_signal_ids'):
                for rid in stats.raw_signal_ids:
                    process_raw_task.delay(str(rid))
        except Exception as e:
            logger.error(f"Ошибка загрузки файла {csv_file}: {e}")
            failed_files.append({'file': str(csv_file), 'error': str(e), 'status': 'failed'})
    return {'status': 'success','directory': directory_path,'equipment_id': equipment_id,'total_files': len(csv_files),'processed_files': len(processed_files),'failed_files': len(failed_files),'details': {'processed': processed_files,'failed': failed_files}}

async def _get_or_create_equipment_from_filename(filename: str) -> str:
    base_name = Path(filename).stem
    equipment_name = base_name.replace('_', ' ').title()
    async with get_async_session() as session:
        result = await session.execute(select(Equipment).where(Equipment.name.ilike(f"%{equipment_name}%")))
        equipment = result.scalar_one_or_none()
        if equipment:
            return str(equipment.id)
        new_equipment = Equipment(name=equipment_name,equipment_type='motor',model=f"Auto-detected from {filename}",location='Unknown',is_active=True,created_at=datetime.utcnow())
        session.add(new_equipment)
        await session.commit()
        await session.refresh(new_equipment)
        return str(new_equipment.id)

@celery_app.task(bind=True)
def process_equipment_workflow(self, equipment_id: str, force_reprocess: bool = False) -> Dict:
    process_raw, _, forecast_trend = _core_tasks()
    try:
        return asyncio.run(_process_equipment_workflow_async(equipment_id, force_reprocess, process_raw, forecast_trend))
    except Exception as exc:
        self.logger.error(f"Ошибка полного цикла для {equipment_id}: {exc}")
        raise

async def _process_equipment_workflow_async(equipment_id: str, force_reprocess: bool, process_raw_task, forecast_trend_task) -> Dict:
    async with get_async_session() as session:
        status_filter = [ProcessingStatus.PENDING]
        if force_reprocess:
            status_filter.extend([ProcessingStatus.COMPLETED, ProcessingStatus.FAILED])
        result = await session.execute(select(RawSignal).where(RawSignal.equipment_id==UUID(equipment_id), RawSignal.processing_status.in_(status_filter)))
        raw_signals = result.scalars().all()
        if not raw_signals:
            return {'status':'no_data','equipment_id':equipment_id,'message':'Нет сигналов для обработки'}
        workflows = [chain(process_raw_task.s(str(rs.id)), _create_anomaly_detection_chord.s()) for rs in raw_signals]
        job = group(workflows)
        workflow_result = job.apply_async()
        forecast_task = forecast_trend_task.apply_async(args=[equipment_id], countdown=300)
        return {'status':'started','equipment_id':equipment_id,'signals_to_process':len(raw_signals),'workflow_job_id':workflow_result.id,'forecast_task_id':forecast_task.id,'force_reprocess':force_reprocess}

@celery_app.task
def _create_anomaly_detection_chord(process_result: Dict) -> str:
    _, detect_anomalies, _ = _core_tasks()
    if process_result.get('status') != 'success':
        return f"Обработка сигнала не удалась: {process_result}"
    feature_ids = process_result.get('feature_ids', [])
    if not feature_ids:
        return "Нет извлеченных признаков для детекции"
    detection_tasks = [detect_anomalies.s(fid) for fid in feature_ids]
    detection_job = group(detection_tasks)
    result = detection_job.apply_async()
    return f"Запущена детекция аномалий для {len(feature_ids)} признаков: {result.id}"

@celery_app.task(bind=True)
def health_check_system(self) -> Dict:
    try:
        return asyncio.run(_health_check_system_async())
    except Exception as exc:
        self.logger.error(f"Ошибка проверки состояния системы: {exc}")
        raise

async def _health_check_system_async() -> Dict:
    async with get_async_session() as session:
        raw_stats = await session.execute(select(RawSignal.processing_status, func.count(RawSignal.id).label('count')).group_by(RawSignal.processing_status))
        raw_signal_stats = {row.processing_status.value: row.count for row in raw_stats}
        total_features = (await session.execute(select(func.count(Feature.id)))).scalar()
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_predictions_count = (await session.execute(select(func.count(Prediction.id)).where(Prediction.created_at>=yesterday))).scalar()
        recent_anomalies_count = (await session.execute(select(func.count(Prediction.id)).where(Prediction.created_at>=yesterday, Prediction.anomaly_detected==True))).scalar()  # noqa: E712
        active_equipment_count = (await session.execute(select(func.count(Equipment.id)).where(Equipment.is_active==True))).scalar()  # noqa: E712
        equipment_with_anomalies_count = (await session.execute(select(func.count(func.distinct(Prediction.equipment_id))).where(Prediction.created_at>=yesterday, Prediction.anomaly_detected==True))).scalar()  # noqa: E712
        return {'status':'healthy','timestamp':datetime.utcnow().isoformat(),'statistics':{'raw_signals':{'by_status':raw_signal_stats,'total':sum(raw_signal_stats.values())},'features':{'total':total_features},'predictions_24h':{'total':recent_predictions_count,'anomalies':recent_anomalies_count,'anomaly_rate':recent_anomalies_count/max(1,recent_predictions_count)},'equipment':{'active':active_equipment_count,'with_recent_anomalies':equipment_with_anomalies_count,'anomaly_equipment_rate':equipment_with_anomalies_count/max(1,active_equipment_count)}},'alerts':_generate_system_alerts(raw_signal_stats,recent_anomalies_count,equipment_with_anomalies_count,active_equipment_count)}

def _generate_system_alerts(raw_signal_stats: Dict, recent_anomalies: int, equipment_with_anomalies: int, active_equipment: int) -> List[Dict]:
    alerts=[]
    failed=raw_signal_stats.get('failed',0)
    total=sum(raw_signal_stats.values())
    if total>0 and failed/total>0.1:
        alerts.append({'level':'warning','type':'processing_failures','message':f'Высокий процент неудачных обработок: {failed}/{total} ({failed/total*100:.1f}%)'})
    if recent_anomalies>50:
        alerts.append({'level':'warning','type':'high_anomaly_count','message':f'Обнаружено много аномалий за последние 24 часа: {recent_anomalies}'})
    if active_equipment>0 and equipment_with_anomalies/active_equipment>0.3:
        alerts.append({'level':'critical','type':'widespread_anomalies','message':f'Аномалии обнаружены на большом количестве оборудования: {equipment_with_anomalies}/{active_equipment} ({equipment_with_anomalies/active_equipment*100:.1f}%)'})
    pending=raw_signal_stats.get('processing',0)
    if pending>100:
        alerts.append({'level':'warning','type':'processing_backlog','message':f'Большое количество сигналов в очереди обработки: {pending}'})
    return alerts

@celery_app.task(bind=True)
def daily_equipment_report(self, equipment_id: Optional[str] = None) -> Dict:
    try:
        return asyncio.run(_daily_equipment_report_async(equipment_id))
    except Exception as exc:
        self.logger.error(f"Ошибка генерации отчета: {exc}")
        raise

async def _daily_equipment_report_async(equipment_id: Optional[str]) -> Dict:
    async with get_async_session() as session:
        yesterday = datetime.utcnow() - timedelta(days=1)
        equipment_filter=[]
        if equipment_id:
            equipment_filter.append(Equipment.id==UUID(equipment_id))
        equipment_result = await session.execute(select(Equipment).where(*equipment_filter, Equipment.is_active==True))  # noqa: E712
        equipment_list = equipment_result.scalars().all()
        reports=[]
        for eq in equipment_list:
            anomalies_count = (await session.execute(select(func.count(Prediction.id)).where(Prediction.equipment_id==eq.id, Prediction.created_at>=yesterday, Prediction.anomaly_detected==True))).scalar()  # noqa: E712
            last_forecast_result = await session.execute(select(Prediction).where(Prediction.equipment_id==eq.id, Prediction.model_name=='rms_trend_forecasting').order_by(Prediction.created_at.desc()).limit(1))
            last_forecast = last_forecast_result.scalar_one_or_none()
            status='normal'
            if anomalies_count>10: status='critical'
            elif anomalies_count>5: status='warning'
            elif anomalies_count>0: status='attention'
            reports.append({'equipment_id':str(eq.id),'equipment_name':eq.name,'status':status,'anomalies_24h':anomalies_count,'last_forecast':{'timestamp': last_forecast.created_at.isoformat() if last_forecast else None,'max_anomaly_probability': last_forecast.confidence if last_forecast else None,'recommendation': last_forecast.prediction_data.get('forecast_summary', {}).get('recommendation') if last_forecast and last_forecast.prediction_data else None}})
        total=len(reports)
        critical=len([r for r in reports if r['status']=='critical'])
        warning=len([r for r in reports if r['status']=='warning'])
        total_anomalies=sum(r['anomalies_24h'] for r in reports)
        return {'report_date': yesterday.date().isoformat(),'generated_at': datetime.utcnow().isoformat(),'summary': {'total_equipment': total,'critical_equipment': critical,'warning_equipment': warning,'normal_equipment': total-critical-warning,'total_anomalies_24h': total_anomalies},'equipment_details': reports,'recommendations': _generate_daily_recommendations(reports)}

def _generate_daily_recommendations(equipment_reports: List[Dict]) -> List[str]:
    rec=[]
    critical=[r for r in equipment_reports if r['status']=='critical']
    warning=[r for r in equipment_reports if r['status']=='warning']
    if critical: rec.append("🔴 КРИТИЧНО: Немедленно проверьте оборудование: "+', '.join([e['equipment_name'] for e in critical]))
    if warning: rec.append("🟡 ВНИМАНИЕ: Запланируйте проверку оборудования: "+', '.join([e['equipment_name'] for e in warning]))
    high=[r for r in equipment_reports if r['anomalies_24h']>20]
    if high: rec.append("📊 Повышенная активность аномалий: "+', '.join([e['equipment_name'] for e in high]))
    if not critical and not warning: rec.append("✅ Все оборудование работает в нормальном режиме")
    return rec

__all__ = ['batch_process_directory','process_equipment_workflow','health_check_system','daily_equipment_report']
