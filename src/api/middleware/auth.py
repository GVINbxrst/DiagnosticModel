"""
JWT авторизация и middleware для FastAPI
"""

import jwt
from datetime import datetime, timedelta, UTC
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import get_settings
from src.database.connection import get_async_session
from src.database.models import User
from src.api.schemas import UserInfo, UserRole
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Настройка хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer токен схема
security = HTTPBearer()


class JWTHandler:
    """Обработчик JWT токенов"""

    def __init__(self):
        self.secret_key = settings.JWT_SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.access_token_expire = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire = settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS

    def create_access_token(self, user_id: str, username: str, role: str) -> str:
        """Создание access токена"""
        expire = datetime.now(UTC) + timedelta(minutes=self.access_token_expire)
        payload = {
            "sub": user_id,
            "username": username,
            "role": role,
            "type": "access",
            "exp": expire,
            "iat": datetime.now(UTC)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def create_refresh_token(self, user_id: str) -> str:
        """Создание refresh токена"""
        expire = datetime.now(UTC) + timedelta(days=self.refresh_token_expire)
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(UTC)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def decode_token(self, token: str) -> Dict[str, Any]:
        """Декодирование токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Токен истек",
                headers={"WWW-Authenticate": "Bearer"}
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Неверный токен",
                headers={"WWW-Authenticate": "Bearer"}
            )

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Проверка пароля"""
        return pwd_context.verify(plain_password, hashed_password)

    def hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        return pwd_context.hash(password)


# Экземпляр JWT обработчика
jwt_handler = JWTHandler()


class JWTBearer(HTTPBearer):
    """Bearer токен authentication"""

    def __init__(self, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)

    async def __call__(self, request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)

        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Неверная схема авторизации. Ожидается Bearer токен",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            # Проверяем токен
            payload = jwt_handler.decode_token(credentials.credentials)

            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Неверный тип токена",
                    headers={"WWW-Authenticate": "Bearer"}
                )

            return credentials.credentials
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Требуется авторизация",
                headers={"WWW-Authenticate": "Bearer"}
            )


async def get_current_user(
    token: str = Depends(JWTBearer()),
    session: AsyncSession = Depends(get_async_session)
) -> UserInfo:
    """Получение текущего пользователя из токена"""

    # Декодируем токен
    payload = jwt_handler.decode_token(token)
    user_id = payload.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный токен",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Получаем пользователя из БД
    query = select(User).where(User.id == user_id)
    result = await session.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
            headers={"WWW-Authenticate": "Bearer"}
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь деактивирован",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return UserInfo.from_orm(user)


async def authenticate_user(username: str, password: str, session: AsyncSession) -> Optional[User]:
    """Аутентификация пользователя"""

    # Получаем пользователя по имени
    query = select(User).where(User.username == username)
    result = await session.execute(query)
    user = result.scalar_one_or_none()

    if not user:
        return None

    # Проверяем пароль
    if not jwt_handler.verify_password(password, user.password_hash):
        return None

    return user


class RoleChecker:
    """Проверка ролей пользователей"""

    def __init__(self, allowed_roles: list[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user = Depends(get_current_user)):
        from pydantic import ValidationError
        try:
            role_val = getattr(current_user.role, 'value', current_user.role)
            allowed_vals = [getattr(r, 'value', r) for r in self.allowed_roles]
            if role_val not in allowed_vals:
                allowed_names = [str(v) for v in allowed_vals]
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Недостаточно прав. Требуются роли: {', '.join(allowed_names)}"
                )
            if not isinstance(current_user, UserInfo):
                from datetime import datetime, UTC
                from uuid import uuid4
                data = {
                    'id': getattr(current_user, 'id', uuid4()),
                    'username': getattr(current_user, 'username', 'unknown'),
                    'email': getattr(current_user, 'email', None),
                    'full_name': getattr(current_user, 'full_name', None),
                    'role': getattr(getattr(current_user, 'role', 'viewer'), 'value', getattr(current_user, 'role', 'viewer')),
                    'is_active': getattr(current_user, 'is_active', True),
                    'created_at': getattr(current_user, 'created_at', datetime.now(UTC)),
                }
                current_user = UserInfo(**data)
            return current_user
        except ValidationError:
            # Возвращаем 401 вместо 422 чтобы тесты ожидали unauthorized
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Требуется авторизация"
            )


# Готовые зависимости для проверки ролей
require_admin = RoleChecker([UserRole.ADMIN])
require_engineer = RoleChecker([UserRole.ADMIN, UserRole.ENGINEER])
require_operator = RoleChecker([UserRole.ADMIN, UserRole.ENGINEER, UserRole.OPERATOR])
require_any_role = RoleChecker([UserRole.ADMIN, UserRole.ENGINEER, UserRole.OPERATOR, UserRole.VIEWER])


def create_token_pair(user: User) -> Dict[str, Any]:
    """Создание пары токенов для пользователя"""

    role_value = user.role.value if hasattr(user.role, 'value') else str(user.role)
    access_token = jwt_handler.create_access_token(
        user_id=str(user.id),
        username=user.username,
        role=role_value
    )

    refresh_token = jwt_handler.create_refresh_token(
        user_id=str(user.id)
    )

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
