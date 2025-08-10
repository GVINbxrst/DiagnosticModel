"""
Endpoint для мониторинга и здоровья системы
Prometheus метрики и health checks
"""
from fastapi import APIRouter, Depends, HTTPException, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import asyncio
import time
from typing import Dict, Any
import psutil
import asyncpg

from src.utils.metrics import get_all_metrics, metrics_collector
from src.utils.logger import get_logger
from src.database.connection import get_database_pool
from src.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter()

settings = get_settings()


@router.get("/metrics")
async def prometheus_metrics():
    """
    Endpoint для Prometheus метрик

    Возвращает все собранные метрики в формате Prometheus
    """
    try:
        metrics_data = get_all_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Ошибка при получении метрик: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения метрик")


@router.get("/health")
async def health_check():
    """
    Базовая проверка здоровья сервиса
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "diagmod-api",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Детальная проверка здоровья всех компонентов системы
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "diagmod-api",
        "version": "1.0.0",
        "components": {}
    }

    # Проверка базы данных
    try:
        db_status = await check_database_health()
        health_status["components"]["database"] = db_status
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Проверка Redis
    try:
        redis_status = await check_redis_health()
        health_status["components"]["redis"] = redis_status
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"

    # Проверка системных ресурсов
    try:
        system_status = check_system_resources()
        health_status["components"]["system"] = system_status

        # Если ресурсы критичны, помечаем как degraded
        if (system_status["cpu_percent"] > 90 or
            system_status["memory_percent"] > 95 or
            system_status["disk_percent"] > 95):
            health_status["status"] = "degraded"

    except Exception as e:
        health_status["components"]["system"] = {
            "status": "unknown",
            "error": str(e)
        }

    # Проверка Worker (Celery)
    try:
        worker_status = await check_worker_health()
        health_status["components"]["worker"] = worker_status
    except Exception as e:
        health_status["components"]["worker"] = {
            "status": "unknown",
            "error": str(e)
        }

    return health_status


@router.get("/health/db")
async def database_health():
    """Проверка здоровья базы данных"""
    try:
        db_status = await check_database_health()
        return db_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")


@router.get("/health/redis")
async def redis_health():
    """Проверка здоровья Redis"""
    try:
        redis_status = await check_redis_health()
        return redis_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unhealthy: {str(e)}")


@router.get("/health/worker")
async def worker_health():
    """Проверка здоровья Celery Worker"""
    try:
        worker_status = await check_worker_health()
        return worker_status
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Worker unhealthy: {str(e)}")


@router.get("/stats")
async def system_stats():
    """
    Статистика системы для мониторинга
    """
    try:
        stats = {
            "system": check_system_resources(),
            "database": await get_database_stats(),
            "api": get_api_stats(),
            "timestamp": time.time()
        }
        return stats
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        raise HTTPException(status_code=500, detail="Ошибка получения статистики")


async def check_database_health() -> Dict[str, Any]:
    """Проверка здоровья базы данных"""
    start_time = time.time()

    try:
        # Получаем пул подключений
        pool = await get_database_pool()

        # Выполняем тестовый запрос
        async with pool.acquire() as connection:
            result = await connection.fetchval("SELECT 1")

        response_time = time.time() - start_time

        return {
            "status": "healthy",
            "response_time_ms": round(response_time * 1000, 2),
            "connections_used": pool.get_size() - pool.get_idle_size(),
            "connections_idle": pool.get_idle_size(),
            "connections_max": pool.get_max_size()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }


async def check_redis_health() -> Dict[str, Any]:
    """Проверка здоровья Redis"""
    import redis.asyncio as redis

    start_time = time.time()

    try:
        # Подключаемся к Redis
        redis_client = redis.from_url(settings.redis_url)

        # Выполняем PING
        result = await redis_client.ping()
        await redis_client.close()

        response_time = time.time() - start_time

        return {
            "status": "healthy" if result else "unhealthy",
            "response_time_ms": round(response_time * 1000, 2),
            "ping_result": result
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }


def check_system_resources() -> Dict[str, Any]:
    """Проверка системных ресурсов"""
    try:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)

        # Память
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Диск
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100

        # Сетевые подключения
        connections = len(psutil.net_connections())

        # Процессы
        process_count = len(psutil.pids())

        return {
            "status": "healthy",
            "cpu_percent": round(cpu_percent, 2),
            "memory_percent": round(memory_percent, 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": round(disk_percent, 2),
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "network_connections": connections,
            "process_count": process_count
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


async def check_worker_health() -> Dict[str, Any]:
    """Проверка здоровья Celery Worker"""
    try:
        from celery import Celery
        from src.worker.config import get_celery_config

        # Создаем экземпляр Celery для проверки
        celery_config = get_celery_config()
        celery_app = Celery('diagmod')
        celery_app.config_from_object(celery_config)

        # Проверяем активные воркеры
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active_tasks = inspect.active()

        if stats:
            worker_count = len(stats)
            total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0

            return {
                "status": "healthy",
                "worker_count": worker_count,
                "active_tasks": total_active_tasks,
                "workers": list(stats.keys())
            }
        else:
            return {
                "status": "unhealthy",
                "error": "No active workers found"
            }

    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e)
        }


async def get_database_stats() -> Dict[str, Any]:
    """Получение статистики базы данных"""
    try:
        pool = await get_database_pool()

        async with pool.acquire() as connection:
            # Статистика таблиц
            tables_stats = await connection.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes
                FROM pg_stat_user_tables 
                WHERE schemaname = 'public'
                ORDER BY n_tup_ins DESC
                LIMIT 10
            """)

            # Размер базы данных
            db_size = await connection.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)

            # Количество подключений
            connections_count = await connection.fetchval("""
                SELECT count(*) FROM pg_stat_activity
            """)

        return {
            "database_size": db_size,
            "active_connections": connections_count,
            "tables_stats": [dict(row) for row in tables_stats]
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def get_api_stats() -> Dict[str, Any]:
    """Получение статистики API"""
    try:
        # Здесь можно добавить статистику из метрик
        return {
            "uptime_seconds": time.time() - metrics_collector.start_time,
            "metrics_collected": True
        }
    except Exception as e:
        return {
            "error": str(e)
        }
