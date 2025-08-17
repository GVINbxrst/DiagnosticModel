"""
Инициализационный модуль для API роутеров
"""

from src.api.routes import upload, anomalies, signals, auth, monitoring, admin_clustering, admin_security

__all__ = [
    'upload',
    'anomalies',
    'signals',
    'auth',
    'monitoring',
    'admin_clustering',
    'admin_security'
]
