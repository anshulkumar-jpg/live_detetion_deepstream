"""Shared fixtures.

Integration tests use a real PostgreSQL (TEST_DATABASE_URL, default
postgresql+asyncpg://vms:vms@localhost:5432/vms_test). GPU telemetry in scheduler
tests is injected as node-heartbeat messages - exactly the input the scheduler
receives from the node agent in production.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.bus.kafka import InMemoryBus
from app.core.config import Settings
from app.db.models import Base

TEST_DB = os.environ.get("TEST_DATABASE_URL", "postgresql+asyncpg://vms:vms@localhost:5432/vms_test")
_KEY = Fernet.generate_key().decode()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test", database_url=TEST_DB, jwt_secret="t" * 40, credential_encryption_key=_KEY,
        internal_service_token="internal-test-token", bootstrap_admin_password="admin-password-123",
        kafka_enabled=False, log_json=False, cameras_per_worker=2, heartbeat_timeout_s=20,
        default_stream_memory_mb=350, gpu_memory_reserve_mb=1024, deepstream_config_dir="/tmp/vms-test-gen",
    )


async def _db_available() -> bool:
    try:
        eng = create_async_engine(TEST_DB)
        async with eng.connect() as c:
            await c.execute(text("select 1"))
        await eng.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
async def engine(settings):
    if not await _db_available():
        pytest.skip("PostgreSQL test database not available")
    eng = create_async_engine(TEST_DB)
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest.fixture
async def sm(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
def bus():
    return InMemoryBus()


def gpu_heartbeat(node="gpu-node-1", index=0, uuid="GPU-TEST-0", mem_total=24576.0, mem_used=2000.0,
                  util=10.0, dec=5.0, enc=0.0, temp=50.0, name="NVIDIA L4") -> dict:
    return {"schema_version": 1, "kind": "node", "node_id": node, "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpus": [{"index": index, "uuid": uuid, "name": name, "driver_version": "550",
                      "memory_total_mb": mem_total, "memory_used_mb": mem_used, "memory_free_mb": mem_total - mem_used,
                      "gpu_util_pct": util, "decoder_util_pct": dec, "encoder_util_pct": enc,
                      "temperature_c": temp, "power_w": 40.0, "processes": []}]}
