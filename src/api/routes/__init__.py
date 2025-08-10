"""
Инициализационный модуль для API роутеров
"""

from src.api.routes import upload, anomalies, signals, auth, monitoring

__all__ = [
    'upload',
    'anomalies',
    'signals',
    'auth',
    'monitoring'
]
