"""Sign-in, sign-out and user management.

Rules
* Signing in is rate limited and locks out: a user name is locked after LOGIN_MAX_FAILURES_PER_USER failures within
  LOGIN_WINDOW_S, a client address after LOGIN_MAX_FAILURES_PER_IP. The counters are in the database, so they hold across
  several API copies. The same answer ("invalid credentials") is given for a wrong password, an unknown user and a
  disabled user, and a password check is always done (against a dummy hash for unknown users), so neither the message
  nor the time reveals which user names exist.
* A login token is only good while the user exists and is active, the token belongs to the user's current token epoch,
  and it was not logged out (see app/api/deps.py principal_from_token). Disabling a user, changing the role or password,
  and "revoke tokens" all raise the epoch and so log the user out everywhere at once.
* Only admins manage users. An admin cannot disable, demote or delete the last active admin (or themselves).
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Principal, admin, audit, client_ip, get_session, viewer
from app.core.config import Settings, get_settings
from app.core.security import Role, create_access_token, dummy_password_hash, hash_password, verify_password
from app.db.models import LoginThrottle, RevokedToken, User, utcnow
from app.schemas.api import PasswordChange, TokenRequest, TokenResponse, UserCreate, UserOut, UserUpdate

router = APIRouter(prefix="/auth", tags=["auth"])


# ------------------------------------------------------------------ sign-in throttling
def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)  # SQLite returns naive times; PostgreSQL returns aware ones
    return dt


async def _retry_after(db: AsyncSession, keys: list[str], now: datetime) -> int:
    rows = (await db.execute(select(LoginThrottle).where(LoginThrottle.key.in_(keys)))).scalars().all()
    waits = [(_aware(r.locked_until) - now).total_seconds() for r in rows if r.locked_until and _aware(r.locked_until) > now]
    return math.ceil(max(waits)) if waits else 0


async def _record_failure(db: AsyncSession, key: str, limit: int, s: Settings, now: datetime) -> None:
    row = (await db.execute(select(LoginThrottle).where(LoginThrottle.key == key).with_for_update())).scalar_one_or_none()
    if row is None:
        row = LoginThrottle(key=key, failures=0, window_started_at=now)
        db.add(row)
    elif (now - _aware(row.window_started_at)).total_seconds() > s.login_window_s:
        row.failures, row.window_started_at, row.locked_until = 0, now, None
    row.failures += 1
    if row.failures >= limit:
        row.locked_until = now + timedelta(seconds=s.login_lockout_s)


async def _forget_old_counters(db: AsyncSession, s: Settings, now: datetime) -> None:
    old = now - timedelta(seconds=s.login_window_s * 4)
    await db.execute(delete(LoginThrottle).where(LoginThrottle.window_started_at < old,
                                                 or_(LoginThrottle.locked_until.is_(None), LoginThrottle.locked_until < now)))


def _too_many(wait: int) -> HTTPException:
    return HTTPException(429, "too many failed sign-in attempts; try again later", headers={"Retry-After": str(max(wait, 1))})


@router.post("/token", response_model=TokenResponse)
async def token(body: TokenRequest, request: Request, db: AsyncSession = Depends(get_session),
                settings: Settings = Depends(get_settings)):
    now = utcnow()
    uname = body.username.strip()
    ukey, ikey = f"u:{uname.lower()[:64]}", f"ip:{client_ip(request)}"
    wait = await _retry_after(db, [ukey, ikey], now)
    if wait:
        raise _too_many(wait)
    u = (await db.execute(select(User).where(User.username == uname))).scalar_one_or_none()
    active = u is not None and u.is_active
    # always one real password check, so an unknown user takes as long as a wrong password (scrypt is CPU heavy: off the loop)
    ok = await asyncio.to_thread(verify_password, body.password, u.password_hash if active else dummy_password_hash())
    if not (active and ok):
        await _forget_old_counters(db, settings, now)
        await _record_failure(db, ukey, settings.login_max_failures_per_user, settings, now)
        await _record_failure(db, ikey, settings.login_max_failures_per_ip, settings, now)
        audit(db, request, Principal(uname[:64] or "anonymous", Role.VIEWER), "auth.login_failed", "user", uname[:64] or None)
        await db.commit()
        raise HTTPException(401, "invalid credentials")
    await db.execute(delete(LoginThrottle).where(LoginThrottle.key == ukey))  # the address counter is NOT reset by a success
    audit(db, request, Principal(u.username, Role(u.role)), "auth.login", "user", u.username)
    await db.commit()
    return TokenResponse(access_token=create_access_token(u.username, Role(u.role), settings, epoch=u.token_epoch),
                         expires_in=settings.jwt_expire_minutes * 60, role=u.role)


@router.post("/logout", status_code=204)
async def logout(request: Request, db: AsyncSession = Depends(get_session), who: Principal = Depends(viewer)):
    """Log out: this token stops working at once."""
    now = utcnow()
    await db.execute(delete(RevokedToken).where(RevokedToken.expires_at < now))  # forget tokens that expired by themselves
    if who.jti:
        db.add(RevokedToken(jti=who.jti, username=who.subject,
                            expires_at=datetime.fromtimestamp(who.exp or int(now.timestamp()), timezone.utc)))
    audit(db, request, who, "auth.logout", "user", who.subject)
    await db.commit()
    return Response(status_code=204)


@router.get("/me")
async def me(who: Principal = Depends(viewer)):
    return {"username": who.subject, "role": who.role.value}


@router.post("/password", status_code=204)
async def change_password(body: PasswordChange, request: Request, db: AsyncSession = Depends(get_session),
                          who: Principal = Depends(viewer), settings: Settings = Depends(get_settings)):
    """Change your own password. All your existing tokens (including this one) stop working: sign in again."""
    now = utcnow()
    ukey = f"u:{who.subject.lower()[:64]}"
    wait = await _retry_after(db, [ukey], now)
    if wait:
        raise _too_many(wait)
    u = await _user(db, who.subject)
    if not await asyncio.to_thread(verify_password, body.current_password, u.password_hash):
        await _record_failure(db, ukey, settings.login_max_failures_per_user, settings, now)
        await db.commit()
        raise HTTPException(401, "current password is wrong")
    u.password_hash = hash_password(body.new_password)
    u.token_epoch += 1
    audit(db, request, who, "user.password_change", "user", u.username)
    await db.commit()
    return Response(status_code=204)


# ------------------------------------------------------------------ user management (admin)
async def _user(db: AsyncSession, username: str) -> User:
    u = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    if u is None:
        raise HTTPException(404, "user not found")
    return u


async def _keep_one_admin(db: AsyncSession, u: User) -> None:
    others = (await db.execute(select(func.count()).select_from(User).where(
        User.role == Role.ADMIN.value, User.is_active.is_(True), User.id != u.id))).scalar_one()
    if not others:
        raise HTTPException(409, "this is the last active admin: create or enable another admin first")


@router.get("/users", response_model=list[UserOut])
async def list_users(db: AsyncSession = Depends(get_session), who: Principal = Depends(admin)):
    now = utcnow()
    locked = {k for (k,) in (await db.execute(select(LoginThrottle.key).where(LoginThrottle.locked_until > now))).all()}
    users = (await db.execute(select(User).order_by(User.username))).scalars().all()
    return [UserOut(username=u.username, role=u.role, is_active=u.is_active, created_at=u.created_at,
                    sites=list(u.sites or []), locked=f"u:{u.username.lower()}" in locked) for u in users]


@router.post("/users", status_code=201)
async def create_user(body: UserCreate, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(admin)):
    if (await db.execute(select(User).where(func.lower(User.username) == body.username.strip().lower()))).scalar_one_or_none():
        raise HTTPException(409, "user exists")
    db.add(User(username=body.username.strip(), password_hash=hash_password(body.password), role=body.role,
                sites=sorted(set(body.sites))))
    audit(db, request, who, "user.create", "user", body.username, {"role": body.role})
    await db.commit()
    return {"username": body.username.strip(), "role": body.role}


@router.patch("/users/{username}", response_model=UserOut)
async def update_user(username: str, body: UserUpdate, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(admin)):
    """Change role, enable/disable, or set a new password. Any of these logs the user out everywhere at once."""
    data = body.model_dump(exclude_unset=True)
    if not data:
        raise HTTPException(422, "nothing to change")
    u = await _user(db, username)
    stops_being_admin = u.role == Role.ADMIN.value and u.is_active and (
        data.get("is_active") is False or ("role" in data and data["role"] != Role.ADMIN.value))
    if stops_being_admin:
        if u.username == who.subject:
            raise HTTPException(409, "you cannot disable or demote your own admin account")
        await _keep_one_admin(db, u)
    changed = []
    if "role" in data and data["role"] != u.role:
        u.role = data["role"]
        changed.append("role")
    if "is_active" in data and data["is_active"] != u.is_active:
        u.is_active = data["is_active"]
        changed.append("is_active")
    if "sites" in data and data["sites"] is not None and sorted(set(data["sites"])) != list(u.sites or []):
        u.sites = sorted(set(data["sites"]))
        changed.append("sites")
    if data.get("password"):
        u.password_hash = hash_password(data["password"])
        changed.append("password")
    if changed:
        u.token_epoch += 1  # every token issued so far stops working
        audit(db, request, who, "user.update", "user", u.username, {"changed": changed})
    await db.commit()
    return UserOut(username=u.username, role=u.role, is_active=u.is_active, created_at=u.created_at,
                   sites=list(u.sites or []))


@router.delete("/users/{username}", status_code=204)
async def delete_user(username: str, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(admin)):
    u = await _user(db, username)
    if u.username == who.subject:
        raise HTTPException(409, "you cannot delete your own account")
    if u.role == Role.ADMIN.value and u.is_active:
        await _keep_one_admin(db, u)
    await db.delete(u)  # tokens of a deleted user are refused at once (the user no longer exists)
    audit(db, request, who, "user.delete", "user", username)
    await db.commit()
    return Response(status_code=204)


@router.post("/users/{username}/revoke-tokens", status_code=204)
async def revoke_tokens(username: str, request: Request, db: AsyncSession = Depends(get_session),
                        who: Principal = Depends(admin)):
    """Log a user out everywhere (for example after a lost laptop). They can sign in again."""
    u = await _user(db, username)
    u.token_epoch += 1
    audit(db, request, who, "user.revoke_tokens", "user", u.username)
    await db.commit()
    return Response(status_code=204)


@router.post("/users/{username}/unlock", status_code=204)
async def unlock_user(username: str, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(admin)):
    """Clear a sign-in lockout (the user locked themselves out with wrong passwords)."""
    await db.execute(delete(LoginThrottle).where(LoginThrottle.key == f"u:{username.strip().lower()[:64]}"))
    audit(db, request, who, "user.unlock", "user", username)
    await db.commit()
    return Response(status_code=204)


async def ensure_bootstrap_admin(db: AsyncSession, settings: Settings) -> bool:
    if (await db.execute(select(User).limit(1))).scalar_one_or_none() is not None:
        return False
    if not settings.bootstrap_admin_password:
        return False
    db.add(User(username=settings.bootstrap_admin_username, password_hash=hash_password(settings.bootstrap_admin_password),
                role=Role.ADMIN.value))
    await db.commit()
    return True
