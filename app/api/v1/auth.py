"""Token issuance and user management."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Principal, admin, audit, get_session
from app.core.config import Settings, get_settings
from app.core.security import Role, create_access_token, hash_password, verify_password
from app.db.models import User
from app.schemas.api import TokenRequest, TokenResponse, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def token(body: TokenRequest, db: AsyncSession = Depends(get_session), settings: Settings = Depends(get_settings)):
    u = (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none()
    if u is None or not u.is_active or not verify_password(body.password, u.password_hash):
        raise HTTPException(401, "invalid credentials")
    return TokenResponse(access_token=create_access_token(u.username, Role(u.role), settings),
                         expires_in=settings.jwt_expire_minutes * 60, role=u.role)


@router.post("/users", status_code=201)
async def create_user(body: UserCreate, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(admin)):
    if (await db.execute(select(User).where(User.username == body.username))).scalar_one_or_none():
        raise HTTPException(409, "user exists")
    db.add(User(username=body.username, password_hash=hash_password(body.password), role=body.role))
    audit(db, request, who, "user.create", "user", body.username, {"role": body.role})
    await db.commit()
    return {"username": body.username, "role": body.role}


async def ensure_bootstrap_admin(db: AsyncSession, settings: Settings) -> bool:
    if (await db.execute(select(User).limit(1))).scalar_one_or_none() is not None:
        return False
    if not settings.bootstrap_admin_password:
        return False
    db.add(User(username=settings.bootstrap_admin_username, password_hash=hash_password(settings.bootstrap_admin_password),
                role=Role.ADMIN.value))
    await db.commit()
    return True
