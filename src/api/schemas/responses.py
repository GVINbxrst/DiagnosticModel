"""
Дополнительные схемы ответов для API
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, UTC


class HealthResponse(BaseModel):
    """Схема ответа для проверки здоровья"""
    status: str = Field(..., description="healthy или unhealthy")
    message: str
    version: str
    timestamp: float
    checks: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Базовая схема успешного ответа"""
    success: bool = True
    message: str
    timestamp: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())


class ErrorResponse(BaseModel):
    """Базовая схема ошибки"""
    error: bool = True
    status_code: int
    message: str
    detail: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
