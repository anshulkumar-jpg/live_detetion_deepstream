"""FastAPI control-plane API."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.v1 import analytics, auth, cameras, internal, models, workers_gpu
from app.bus.kafka import InMemoryBus
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, request_id_var
from app.core.metrics import API
from app.db.session import db_ping, get_sessionmaker, init_engine, wait_for_db
from app.services.mediamtx import StreamManager
from app.services.scheduler import Scheduler

log = logging.getLogger(__name__)


def build_bus(settings: Settings):
    if not settings.kafka_enabled:
        log.warning("KAFKA_ENABLED=false: using in-process bus (commands will NOT reach workers)")
        return InMemoryBus()
    from app.bus.kafka import KafkaBus

    return KafkaBus(settings, client_id=settings.service_name)


def create_app(settings: Settings | None = None, bus=None, streams=None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(s.service_name, s.log_level, s.log_json)
        if s.environment == "production":
            s.validate_for_production()
        init_engine(s)
        await wait_for_db()
        async with get_sessionmaker()() as db:
            if await auth.ensure_bootstrap_admin(db, s):
                log.info("bootstrap admin user created")
        app.state.bus = bus or build_bus(s)
        app.state.streams = streams or StreamManager(s)
        app.state.scheduler = Scheduler(get_sessionmaker(), app.state.bus, s, source=s.service_name,
                                        stream_manager=app.state.streams)
        yield
        app.state.bus.close()

    app = FastAPI(
        title="DeepStream VMS Backend",
        version="1.0.0",
        description="Control-plane API for the DeepStream multi-camera analytics platform. "
                    "Authenticate with POST /api/v1/auth/token and send `Authorization: Bearer <token>`.",
        lifespan=lifespan,
    )
    # every request-scoped dependency sees the same settings object the app was built with
    app.dependency_overrides[get_settings] = lambda: s
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            API.requests.labels(request.method, path, str(status_code)).inc()
            API.latency.labels(request.method, path).observe(time.perf_counter() - start)
            request_id_var.reset(token)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": request_id_var.get()})

    prefix = "/api/v1"
    for r in (auth.router, cameras.router, models.router, workers_gpu.workers, workers_gpu.gpu, analytics.router):
        app.include_router(r, prefix=prefix)
    app.include_router(internal.router)

    @app.get("/health", tags=["system"])
    async def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    async def ready(request: Request):
        checks = {"database": await db_ping()}
        b = getattr(request.app.state, "bus", None)
        checks["bus"] = b is not None
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503, content={"ready": ok, "checks": checks})

    @app.get("/metrics", tags=["system"])
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
