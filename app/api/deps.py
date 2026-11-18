"""Shared FastAPI dependencies: DB session, auth/RBAC, audit logging, bus."""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import request_id_var
from app.core.security import ROLE_RANK, Role, decode_token, internal_token_valid
from app.db.models import AuditLog
from app.db.session import get_sessionmaker

bearer = HTTPBearer(auto_error=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@dataclass
class Principal:
    subject: str
    role: Role


def get_principal(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    # internal services authenticate with a shared service token
    internal = request.headers.get("x-internal-token")
    if internal is not None:
        if internal_token_valid(internal, settings):
            return Principal("internal-service", Role.ADMIN)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        claims = decode_token(creds.credentials, settings)
        return Principal(claims["sub"], Role(claims["role"]))
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token", headers={"WWW-Authenticate": "Bearer"})


def require(role: Role):
    def dep(p: Principal = Depends(get_principal)) -> Principal:
        if ROLE_RANK[p.role] < ROLE_RANK[role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role '{role.value}'")
        return p
    return dep


viewer = require(Role.VIEWER)
operator = require(Role.OPERATOR)
admin = require(Role.ADMIN)


def audit(db: AsyncSession, request: Request, who: Principal, action: str, rtype: str, rid: str | None, details: dict | None = None) -> None:
    db.add(AuditLog(
        actor=who.subject, action=action, resource_type=rtype, resource_id=rid,
        details=details or {}, request_id=request_id_var.get(),
        client_ip=request.client.host if request.client else None,
    ))


def get_bus(request: Request):
    return request.app.state.bus


def get_scheduler(request: Request):
    return request.app.state.scheduler


def get_streams(request: Request):
    return request.app.state.streams
