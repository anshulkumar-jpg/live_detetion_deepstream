"""Worker and GPU resource endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import Principal, admin, audit, get_scheduler, get_session, operator, viewer
from app.core.config import Settings, get_settings
from app.db.models import (
    CameraStatus,
    Camera,
    CapacityProfile,
    GpuDevice,
    GpuResourceMetric,
    Worker,
    WorkerCameraAssignment,
    WorkerState,
)
from app.schemas.api import CapacityProfileIn, WorkerOut
from app.services.gpu import StreamCost, admit
from app.services.model_router import RoutingError

workers = APIRouter(prefix="/workers", tags=["workers"])
gpu = APIRouter(prefix="/gpu", tags=["gpu"])


async def _cams(db: AsyncSession, wid: str) -> list[str]:
    return list((await db.execute(select(WorkerCameraAssignment.camera_id).where(and_(
        WorkerCameraAssignment.worker_id == wid, WorkerCameraAssignment.released_at.is_(None))))).scalars())


def _wout(w: Worker, cams: list[str]) -> WorkerOut:
    return WorkerOut.model_validate({**{c.key: getattr(w, c.key) for c in Worker.__table__.columns}, "camera_ids": cams})


@workers.get("", response_model=list[WorkerOut])
async def list_workers(db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer),
                       state: WorkerState | None = None, include_stopped: bool = False,
                       limit: int = Query(500, ge=1, le=2000)):
    q = select(Worker)
    if state is not None:
        q = q.where(Worker.state == state)
    elif not include_stopped:
        q = q.where(Worker.state.notin_([WorkerState.STOPPED, WorkerState.FAILED]))
    ws = list((await db.execute(q.order_by(Worker.created_at.desc()).limit(limit))).scalars())
    rows = (await db.execute(select(WorkerCameraAssignment.worker_id, WorkerCameraAssignment.camera_id).where(and_(
        WorkerCameraAssignment.worker_id.in_([w.id for w in ws]), WorkerCameraAssignment.released_at.is_(None))))).all()
    by = {}
    for wid, cid in rows:
        by.setdefault(wid, []).append(cid)
    return [_wout(w, by.get(w.id, [])) for w in ws]


@workers.get("/{worker_id}", response_model=WorkerOut)
async def get_worker(worker_id: str, db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer)):
    w = await db.get(Worker, worker_id)
    if w is None:
        raise HTTPException(404, "worker not found")
    return _wout(w, await _cams(db, w.id))


@workers.get("/{worker_id}/metrics")
async def worker_metrics(worker_id: str, db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer)):
    w = await db.get(Worker, worker_id)
    if w is None:
        raise HTTPException(404, "worker not found")
    return {"worker_id": w.id, "state": w.state.value, "last_heartbeat_at": w.last_heartbeat_at,
            "metrics": w.metrics or {}}


@workers.post("/{worker_id}/restart", status_code=202)
async def restart_worker(worker_id: str, request: Request, db: AsyncSession = Depends(get_session),
                         who: Principal = Depends(operator), sched=Depends(get_scheduler)):
    try:
        w = await sched.request_worker_restart(db, worker_id)
    except KeyError:
        raise HTTPException(404, "worker not found")
    except (ValueError, RoutingError) as e:
        raise HTTPException(409, str(e))
    audit(db, request, who, "worker.restart", "worker", worker_id)
    await db.commit()
    return {"worker_id": w.id, "state": w.state.value, "spec_generation": w.spec_generation}


@workers.post("/{worker_id}/stop", status_code=202)
async def stop_worker(worker_id: str, request: Request, db: AsyncSession = Depends(get_session),
                      who: Principal = Depends(operator), sched=Depends(get_scheduler)):
    try:
        w = await sched.request_worker_stop(db, worker_id)
    except KeyError:
        raise HTTPException(404, "worker not found")
    audit(db, request, who, "worker.stop", "worker", worker_id)
    await db.commit()
    return {"worker_id": w.id, "state": w.state.value,
            "note": "cameras still desired RUNNING will be rescheduled onto other workers"}


# ------------------------------------------------------------------ GPU
@gpu.get("/status")
async def gpu_status(db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer),
                     settings: Settings = Depends(get_settings)):
    now = datetime.now(timezone.utc)
    out = []
    for g in (await db.execute(select(GpuDevice).order_by(GpuDevice.node_id, GpuDevice.gpu_index))).scalars():
        seen = g.last_seen_at.replace(tzinfo=g.last_seen_at.tzinfo or timezone.utc) if g.last_seen_at else None
        age = (now - seen).total_seconds() if seen else None
        out.append({"id": g.id, "node_id": g.node_id, "index": g.gpu_index, "uuid": g.uuid, "name": g.name,
                    "enabled": g.enabled, "healthy": g.healthy, "metrics_age_s": age,
                    "stale": age is None or age > settings.gpu_metrics_stale_s, "driver_version": g.driver_version})
    return {"gpus": out, "count": len(out)}


@gpu.get("/metrics")
async def gpu_metrics(db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer),
                      minutes: int = Query(15, ge=1, le=1440)):
    since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    latest = {g.id: g.latest_metrics for g in (await db.execute(select(GpuDevice))).scalars()}
    hist = (await db.execute(select(GpuResourceMetric).where(GpuResourceMetric.ts >= since)
                             .order_by(GpuResourceMetric.ts))).scalars()
    series: dict[int, list] = {}
    for m in hist:
        series.setdefault(m.gpu_device_id, []).append({
            "ts": m.ts, "gpu_util_pct": m.gpu_util_pct, "memory_used_mb": m.memory_used_mb,
            "decoder_util_pct": m.decoder_util_pct, "encoder_util_pct": m.encoder_util_pct,
            "temperature_c": m.temperature_c, "power_w": m.power_w})
    return {"latest": latest, "history": series}


@gpu.get("/resources")
async def gpu_resources(db: AsyncSession = Depends(get_session), _: Principal = Depends(viewer),
                        settings: Settings = Depends(get_settings), sched=Depends(get_scheduler)):
    """Capacity view: streams per GPU, pending cameras, and the live admission decision."""
    ws = {w.id: w for w in (await db.execute(select(Worker))).scalars()}
    active = await sched._active_assignments(db)
    snaps = await sched._gpu_snapshots(db, active, ws)
    pending = (await db.execute(select(Camera.id, Camera.status_reason).where(
        Camera.status == CameraStatus.PENDING_CAPACITY).limit(100))).all()
    pending_n = (await db.execute(select(func.count()).select_from(Camera).where(
        Camera.status == CameraStatus.PENDING_CAPACITY))).scalar()
    gpus = []
    for s in snaps:
        d = admit(s, StreamCost(memory_mb=settings.default_stream_memory_mb), settings)
        gpus.append({"gpu_device_id": s.device_id, "name": s.name, "active_streams": s.active_streams,
                     "unconfirmed_streams": s.unconfirmed_streams, "avg_latency_ms": s.avg_inference_latency_ms,
                     "accepting_new_streams": d.admitted, "admission_reason": d.reason, "checks": d.checks,
                     "workers": [w.id for w in ws.values() if w.gpu_device_id == s.device_id
                                 and w.state not in (WorkerState.STOPPED, WorkerState.FAILED)]})
    profiles = [{"id": p.id, "gpu_name": p.gpu_name, "pipeline_signature": p.pipeline_signature,
                 "max_streams": p.max_streams, "memory_mb_per_stream": p.memory_mb_per_stream,
                 "avg_fps_per_stream": p.avg_fps_per_stream, "measured_at": p.measured_at}
                for p in (await db.execute(select(CapacityProfile).order_by(CapacityProfile.measured_at.desc()))).scalars()]
    return {"thresholds": {k: getattr(settings, k) for k in (
                "gpu_max_utilization_pct", "gpu_max_memory_pct", "gpu_max_decoder_pct", "gpu_max_encoder_pct",
                "gpu_max_temperature_c", "gpu_max_inference_latency_ms", "gpu_memory_reserve_mb",
                "default_stream_memory_mb", "default_max_streams_per_gpu", "cameras_per_worker")},
            "gpus": gpus, "pending_capacity": {"count": pending_n, "cameras": [dict(id=i, reason=r) for i, r in pending]},
            "capacity_profiles": profiles}


@gpu.post("/capacity-profiles", status_code=201)
async def add_capacity_profile(body: CapacityProfileIn, request: Request, db: AsyncSession = Depends(get_session),
                               who: Principal = Depends(admin)):
    """Record benchmark-measured capacity (scripts/benchmark.py posts here)."""
    p = CapacityProfile(**body.model_dump())
    db.add(p)
    audit(db, request, who, "gpu.capacity_profile", "capacity_profile", body.pipeline_signature, body.model_dump())
    await db.commit()
    return {"id": p.id}
