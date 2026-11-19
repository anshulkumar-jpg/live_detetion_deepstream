"""Shared FastAPI dependencies: DB session, auth/RBAC, audit logging, bus."""
from __future__ import annotations

import ipaddress
from collections.abc import AsyncIterator
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import request_id_var
from app.core.security import ROLE_RANK, Role, decode_token, decode_video_token
from app.db.models import AuditLog, RevokedToken, User
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
    jti: str | None = None  # id of the token this request used (needed to log it out)
    exp: int | None = None
    sites: tuple[str, ...] = ()  # sites this user may access; empty = all

    def can_see_site(self, site: str) -> bool:
        return not self.sites or site in self.sites


class AuthError(Exception):
    """The login token is not (or no longer) valid."""


def _is_private(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False


def client_ip(request: Request) -> str:
    """The real client address. We sit behind our own gateway (Envoy), which appends the address it saw as the LAST entry
    of X-Forwarded-For; that header is only trusted when the request really came from a private (gateway) address."""
    peer = request.client.host if request.client else ""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd and _is_private(peer):
        return fwd.split(",")[-1].strip()[:64] or peer
    return peer


async def principal_from_token(db: AsyncSession, token: str, settings: Settings) -> Principal:
    """Check a login token against the database, not just its signature.

    The user must still exist and be active, the token must belong to the user's current token epoch (disabling a user,
    changing role or password, and the admin "revoke tokens" action all raise it), and it must not have been logged out.
    The role comes from the database, so a demotion takes effect on the very next request.
    """
    try:
        claims = decode_token(token, settings)
        username, jti = str(claims["sub"]), str(claims["jti"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise AuthError("invalid or expired token")
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthError("account disabled or removed")
    try:
        token_epoch = int(claims.get("te", 0))
    except (TypeError, ValueError):
        raise AuthError("invalid or expired token")
    if token_epoch != int(user.token_epoch or 0):
        raise AuthError("token revoked")
    if (await db.execute(select(RevokedToken.jti).where(RevokedToken.jti == jti))).first() is not None:
        raise AuthError("token revoked")
    return Principal(user.username, Role(user.role), jti, int(claims.get("exp", 0)), tuple(user.sites or ()))


async def video_principal_from_token(db: AsyncSession, token: str, camera_id: str, settings: Settings) -> Principal:
    """Check a camera-scoped video token: right audience, right camera, user still active, token epoch current.
    A login (API) token is refused here on purpose: it has the wrong audience."""
    try:
        claims = decode_video_token(token, settings)
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise AuthError("invalid or expired video token")
    if claims.get("cam") != camera_id:
        raise AuthError("video token is for a different camera")
    user = (await db.execute(select(User).where(User.username == str(claims["sub"])))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthError("account disabled or removed")
    try:
        if int(claims.get("te", 0)) != int(user.token_epoch or 0):
            raise AuthError("token revoked")
    except (TypeError, ValueError):
        raise AuthError("invalid or expired video token")
    return Principal(user.username, Role(user.role), str(claims.get("jti")), int(claims.get("exp", 0)), tuple(user.sites or ()))


async def get_principal(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
    db: AsyncSession = Depends(get_session),
) -> Principal:
    # Only a signed user token (JWT) opens the API. There is deliberately no "service token" path: services talk to
    # each other through Kafka and the database, and the only shared secret left (the video publish token) can
    # publish video and nothing else.
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return await principal_from_token(db, creds.credentials, settings)
    except AuthError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e), headers={"WWW-Authenticate": "Bearer"})


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
        client_ip=client_ip(request),
    ))


def get_bus(request: Request):
    return request.app.state.bus


def get_scheduler(request: Request):
    return request.app.state.scheduler


def get_streams(request: Request):
    return request.app.state.streams
