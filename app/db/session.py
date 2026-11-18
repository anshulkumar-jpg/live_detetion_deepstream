"""Async engine/session management with connection recovery (pool_pre_ping)."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings

log = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings | None = None, url: str | None = None) -> AsyncEngine:
    global _engine, _sessionmaker
    s = settings or get_settings()
    db_url = url or s.database_url
    kwargs: dict = {"pool_pre_ping": True}
    if not db_url.startswith("sqlite"):
        kwargs.update(pool_size=s.db_pool_size, max_overflow=s.db_max_overflow, pool_recycle=1800)
    _engine = create_async_engine(db_url, **kwargs)
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        init_engine()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope: commit on success, rollback on error."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def wait_for_db(max_wait_s: float = 120.0) -> None:
    """Block until the database answers, with exponential backoff (startup recovery)."""
    delay, waited = 0.5, 0.0
    while True:
        try:
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
            return
        except (OperationalError, DBAPIError, OSError) as e:
            if waited >= max_wait_s:
                raise
            log.warning("database not ready (%s); retrying in %.1fs", type(e).__name__, delay)
            await asyncio.sleep(delay)
            waited += delay
            delay = min(delay * 2, 10.0)


async def db_ping() -> bool:
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
