"""
Роутер для авторизации пользователей с расширенной безопасностью
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.middleware.auth import authenticate_user, get_current_user
from src.api.middleware.security import enhanced_jwt_handler, audit_logger, AuditActionType, AuditResult
from src.api.schemas import LoginRequest, TokenResponse, RefreshTokenRequest, UserInfo
from src.database.connection import get_async_session
from src.database.models import User
from src.utils.logger import get_logger
from src.utils.metrics import observe_latency

router = APIRouter()
logger = get_logger(__name__)


@router.post("/login", response_model=TokenResponse)
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/login'})
async def login(
    credentials: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_async_session)
):
    """Авторизация пользователя с audit-логированием"""

    # Аутентификация
    user = await authenticate_user(credentials.username, credentials.password, session)

    if not user:
        # Логируем неудачную попытку входа
        await audit_logger.log_action(
            user_id=user.id if user else None,
            username=credentials.username,
            user_role="unknown",
            action_type=AuditActionType.FAILED_LOGIN,
            result=AuditResult.FAILURE,
            request=request,
            action_description=f"Failed login attempt for user {credentials.username}"
        )

        logger.warning(f"Неудачная попытка входа для пользователя: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        await audit_logger.log_action(
            user_id=user.id,
            username=user.username,
            user_role=getattr(user.role, 'value', user.role),
            action_type=AuditActionType.FAILED_LOGIN,
            result=AuditResult.DENIED,
            request=request,
            action_description="Login attempt for deactivated account"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Аккаунт деактивирован"
        )

    # Создаем сессию с токенами (логирование внутри enhanced_jwt_handler)
    token_data = await enhanced_jwt_handler.create_session(user, request)

    logger.info(f"Успешная авторизация пользователя: {user.username} (роль: {getattr(user.role, 'value', user.role)})")
    
    return TokenResponse(**token_data)


@router.post("/refresh", response_model=TokenResponse)
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/refresh'})
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    request: Request
):
    """Обновление access токена с audit-логированием"""

    try:
        # Обновляем сессию (логирование внутри enhanced_jwt_handler)
        token_data = await enhanced_jwt_handler.refresh_session(
            refresh_request.refresh_token,
            request
        )

        return TokenResponse(**token_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обновления токена: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный refresh токен"
        )


@router.post("/logout")
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/logout'})
async def logout(
    request: Request,
    current_user: UserInfo = Depends(get_current_user)
):
    """Выход из системы с отзывом сессии"""

    try:
        # Извлекаем session_id из токена
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = enhanced_jwt_handler._decode_token(token)
            session_id = payload.get("session_id")

            if session_id:
                # Отзываем сессию
                await enhanced_jwt_handler.revoke_session(session_id, "User logout")

        # Логируем выход
        await audit_logger.log_action(
            user_id=current_user.id,
            username=current_user.username,
            user_role=getattr(current_user.role, 'value', current_user.role),
            action_type=AuditActionType.LOGOUT,
            result=AuditResult.SUCCESS,
            request=request
        )

        return {"message": "Successfully logged out"}

    except Exception as e:
        logger.error(f"Ошибка выхода из системы: {e}")
        return {"message": "Logout completed"}


@router.get("/me", response_model=UserInfo)
@observe_latency('api_request_duration_seconds', labels={'method':'GET','endpoint':'/me'})
async def get_current_user_info(
    current_user: UserInfo = Depends(get_current_user)
):
    """Получение информации о текущем пользователе"""
    return current_user


@router.post("/revoke-session")
@observe_latency('api_request_duration_seconds', labels={'method':'POST','endpoint':'/revoke-session'})
async def revoke_session(
    session_id: str,
    request: Request,
    current_user: UserInfo = Depends(get_current_user)
):
    """Отзыв конкретной сессии (для администраторов)"""

    if current_user.role not in ["admin"]:
        await audit_logger.log_action(
            user_id=current_user.id,
            username=current_user.username,
            user_role=getattr(current_user.role, 'value', current_user.role),
            action_type=AuditActionType.PERMISSION_DENIED,
            result=AuditResult.DENIED,
            request=request,
            action_description="Attempted to revoke session without admin privileges"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Недостаточно прав для отзыва сессий"
        )

    try:
        await enhanced_jwt_handler.revoke_session(
            session_id,
            f"Revoked by admin {current_user.username}"
        )

        await audit_logger.log_action(
            user_id=current_user.id,
            username=current_user.username,
            user_role=current_user.role.value,
            action_type=AuditActionType.ADMIN_ACTION,
            result=AuditResult.SUCCESS,
            request=request,
            action_description=f"Revoked session {session_id}"
        )

        return {"message": f"Session {session_id} revoked successfully"}

    except Exception as e:
        logger.error(f"Ошибка отзыва сессии: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка отзыва сессии"
        )
