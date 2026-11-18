"""Worker Scheduler: database-driven reconciliation of desired vs actual state.

Model
-----
* A *logical worker* is one DeepStream pipeline group holding up to
  ``cameras_per_worker`` cameras (default 2) that share an identical inference
  plan (same pipeline signature). The node agent runs each logical worker as an
  isolated OS process, so a crash only affects that worker's cameras.
* Desired state lives in PostgreSQL (camera.desired_state, worker.desired_state).
  The scheduler converges actual state toward it every ``scheduler_interval_s``
  and on every event, so it recovers from lost messages, crashes and restarts.
* New cameras are added to an existing worker with a free slot via ADD_SOURCE
  (other cameras keep running) or to a new worker via START_WORKER.
* Before any assignment the GPU Resource Manager's admission control runs; if no
  GPU can take the stream the camera is parked in PENDING_CAPACITY with a reason
  and retried on later passes.
"""
from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.bus.schemas import Envelope, PipelineCommandType, PipelineEventType, Topics
from app.core.config import Settings
from app.core.metrics import PIPE, SCHED
from app.db.models import (
    AssignmentState,
    Camera,
    CameraModelAssignment,
    CameraStatus,
    CapacityProfile,
    DesiredState,
    GpuDevice,
    GpuResourceMetric,
    Model,
    ModelVersion,
    ModelVersionStatus,
    PipelineSession,
    ProcessedCommand,
    StreamMode,
    Worker,
    WorkerCameraAssignment,
    WorkerState,
)
from app.services.gpu import GpuSnapshot, StreamCost, admit
from app.services.model_router import PipelinePlan, RoutingError, build_plan

log = logging.getLogger(__name__)

LIVE_WORKER_STATES = (WorkerState.CREATED, WorkerState.STARTING, WorkerState.RUNNING, WorkerState.DEGRADED)
SCHEDULER_LOCK_ID = 727_001


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def stream_path(camera_id: str, kind: str = "processed") -> str:
    return f"cam/{camera_id}/{kind}"


class Scheduler:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], bus, settings: Settings, source: str = "scheduler",
                 stream_manager=None):
        self.sm = sessionmaker
        self.bus = bus
        self.s = settings
        self.source = source
        self.streams = stream_manager  # optional MediaMTX manager for RAW paths
        self._last_gpu_sample: dict[int, datetime] = {}
        self._worker_latency: dict[str, float] = {}

    # =========================================================== commands
    def _command(self, node_id: str, ctype: PipelineCommandType, payload: dict[str, Any]) -> str:
        env = Envelope(event_type=ctype.value, source=self.source, payload={"node_id": node_id, **payload})
        self.bus.publish_reliable(Topics.PIPELINE_COMMANDS, env.model_dump(), key=node_id)
        SCHED.commands.labels(ctype.value).inc()
        return env.event_id

    def _camera_status(self, cam: Camera, status: CameraStatus, reason: str | None = None) -> None:
        if cam.status == status and cam.status_reason == reason:
            return
        cam.status = status
        cam.status_reason = reason
        cam.status_changed_at = utcnow()
        self.bus.publish(Topics.CAMERA_STATUS, Envelope(
            event_type="CAMERA_STATUS", source=self.source,
            payload={"camera_id": cam.id, "status": status.value, "reason": reason}).model_dump(), key=cam.id)

    # =========================================================== helpers
    async def _load_routing_catalog(self, db: AsyncSession) -> tuple[dict[str, Model], dict[int, ModelVersion]]:
        models = {m.id: m for m in (await db.execute(select(Model))).scalars()}
        versions = {v.id: v for v in (await db.execute(select(ModelVersion))).scalars()}
        return models, versions

    async def _assignments_for(self, db: AsyncSession, camera_ids: list[str]) -> dict[str, list[CameraModelAssignment]]:
        out: dict[str, list[CameraModelAssignment]] = defaultdict(list)
        if not camera_ids:
            return out
        rows = (await db.execute(select(CameraModelAssignment).where(CameraModelAssignment.camera_id.in_(camera_ids)))).scalars()
        for a in rows:
            out[a.camera_id].append(a)
        return out

    async def _active_assignments(self, db: AsyncSession) -> list[WorkerCameraAssignment]:
        return list((await db.execute(select(WorkerCameraAssignment).where(WorkerCameraAssignment.released_at.is_(None)))).scalars())

    async def _gpu_snapshots(self, db: AsyncSession, active: list[WorkerCameraAssignment], workers: dict[str, Worker]) -> list[GpuSnapshot]:
        gpus = list((await db.execute(select(GpuDevice).where(GpuDevice.enabled.is_(True)))).scalars())
        per_gpu = defaultdict(int)
        unconfirmed = defaultdict(int)
        lat_sum, lat_n = defaultdict(float), defaultdict(int)
        for a in active:
            w = workers.get(a.worker_id)
            if w is None or w.gpu_device_id is None:
                continue
            per_gpu[w.gpu_device_id] += 1
            if a.state != AssignmentState.ACTIVE:
                unconfirmed[w.gpu_device_id] += 1
        for w in workers.values():
            lat = (w.metrics or {}).get("avg_latency_ms")
            if w.gpu_device_id is not None and lat is not None and w.state in LIVE_WORKER_STATES:
                lat_sum[w.gpu_device_id] += float(lat)
                lat_n[w.gpu_device_id] += 1
        now = utcnow()
        snaps = []
        for g in gpus:
            age = (now - _aware(g.last_seen_at)).total_seconds() if g.last_seen_at else 1e9
            snaps.append(GpuSnapshot(
                device_id=g.id, name=g.name or "unknown", healthy=g.healthy, metrics=g.latest_metrics or {},
                last_seen_age_s=age, active_streams=per_gpu[g.id], unconfirmed_streams=unconfirmed[g.id],
                avg_inference_latency_ms=(lat_sum[g.id] / lat_n[g.id]) if lat_n[g.id] else None,
            ))
        return snaps

    async def _stream_cost(self, db: AsyncSession, gpu_name: str, plan: PipelinePlan, versions: dict[int, ModelVersion]) -> StreamCost:
        prof = (await db.execute(
            select(CapacityProfile).where(and_(CapacityProfile.gpu_name == gpu_name,
                                               CapacityProfile.pipeline_signature == plan.signature))
            .order_by(CapacityProfile.measured_at.desc()).limit(1))).scalar_one_or_none()
        if prof is not None:
            return StreamCost(memory_mb=prof.memory_mb_per_stream or self.s.default_stream_memory_mb,
                              max_streams=prof.max_streams, source=f"benchmark#{prof.id}")
        # no benchmark: sum measured model memory if known, else configured default
        mem = 0.0
        for st in plan.stages:
            v = versions.get(st.model_version_id)
            if v and v.est_memory_mb:
                mem += v.est_memory_mb / max(1, self.s.cameras_per_worker)
        return StreamCost(memory_mb=max(mem, float(self.s.default_stream_memory_mb)), source="default")

    def build_source_spec(self, cam: Camera) -> dict[str, Any]:
        return {
            "camera_id": cam.id,
            "uri": cam.rtsp_url,
            "username_enc": cam.username_enc,
            "password_enc": cam.password_enc,
            "camera_generation": cam.config_generation,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
            "annotated_output": cam.stream_mode in (StreamMode.ANNOTATED, StreamMode.BOTH),
            "output_path": stream_path(cam.id, "processed"),
        }

    def build_worker_spec(self, w: Worker, plan: PipelinePlan, gpu_index: int, sources: list[dict]) -> dict[str, Any]:
        return {
            "worker_id": w.id,
            "spec_generation": w.spec_generation,
            "gpu_index": gpu_index,
            "capacity": w.capacity,
            "plan": plan.to_dict(),
            "sources": sources,
        }

    # =========================================================== leader lock
    async def try_acquire_leadership(self, conn) -> bool:
        if conn.dialect.name != "postgresql":
            return True
        res = await conn.execute(text("SELECT pg_try_advisory_lock(:id)"), {"id": SCHEDULER_LOCK_ID})
        return bool(res.scalar())

    # =========================================================== reconcile
    async def reconcile_once(self) -> dict[str, int]:
        stats = defaultdict(int)
        async with self.sm() as db:
            try:
                await self._detect_failed_workers(db, stats)
                await self._stop_requested(db, stats)
                await self._start_requested(db, stats)
                await self._retire_empty_workers(db, stats)
                await db.commit()
            except Exception:
                await db.rollback()
                raise
        await self._update_gauges()
        return dict(stats)

    async def _detect_failed_workers(self, db: AsyncSession, stats) -> None:
        now = utcnow()
        workers = (await db.execute(select(Worker).where(Worker.state.in_(
            [WorkerState.STARTING, WorkerState.RUNNING, WorkerState.DEGRADED, WorkerState.STOPPING])))).scalars()
        for w in workers:
            hb = _aware(w.last_heartbeat_at)
            changed = _aware(w.state_changed_at) or _aware(w.created_at) or now
            if w.state == WorkerState.STOPPING:
                if (now - changed).total_seconds() > self.s.worker_start_timeout_s:
                    await self._finalize_worker(db, w, WorkerState.STOPPED, "stop timeout")
                continue
            if w.state == WorkerState.STARTING and hb is None:
                stale = (now - changed).total_seconds() > self.s.worker_start_timeout_s
                reason = "worker did not report within start timeout"
            else:
                stale = hb is None or (now - hb).total_seconds() > self.s.heartbeat_timeout_s
                reason = "heartbeat timeout"
            if stale:
                log.warning("worker failed: %s", reason, extra={"worker_id": w.id})
                await self._finalize_worker(db, w, WorkerState.FAILED, reason)
                # make sure a zombie process is torn down; healthy workers are untouched
                self._command(w.node_id, PipelineCommandType.STOP_WORKER, {"worker_id": w.id, "reason": reason})
                stats["workers_failed"] += 1
                SCHED.worker_failures.inc()

    async def _finalize_worker(self, db: AsyncSession, w: Worker, state: WorkerState, reason: str) -> None:
        w.state = state
        w.state_changed_at = utcnow()
        w.stopped_at = utcnow()
        if state == WorkerState.FAILED:
            w.last_error = reason
            w.health = "failed"
        rows = (await db.execute(select(WorkerCameraAssignment).where(and_(
            WorkerCameraAssignment.worker_id == w.id, WorkerCameraAssignment.released_at.is_(None))))).scalars()
        for a in rows:
            a.state = AssignmentState.RELEASED
            a.released_at = utcnow()
            a.reason = f"worker {state.value.lower()}: {reason}"[:255]
            cam = await db.get(Camera, a.camera_id)
            await self._end_pipeline_session(db, a.camera_id, w.id, a.reason)
            if cam is None:
                continue
            if cam.desired_state == DesiredState.RUNNING:
                # recovery: the camera will be re-admitted on the next pass
                self._camera_status(cam, CameraStatus.SCHEDULED, f"recovering after {reason}")
            else:
                self._camera_status(cam, CameraStatus.STOPPED, None)

    async def _end_pipeline_session(self, db: AsyncSession, camera_id: str, worker_id: str, reason: str) -> None:
        await db.execute(update(PipelineSession).where(and_(
            PipelineSession.camera_id == camera_id, PipelineSession.worker_id == worker_id,
            PipelineSession.ended_at.is_(None))).values(ended_at=utcnow(), end_reason=reason[:255]))

    async def _stop_requested(self, db: AsyncSession, stats) -> None:
        """Cameras that must leave their worker: desired STOPPED, or config generation changed."""
        rows = (await db.execute(
            select(WorkerCameraAssignment, Camera, Worker)
            .join(Camera, Camera.id == WorkerCameraAssignment.camera_id)
            .join(Worker, Worker.id == WorkerCameraAssignment.worker_id)
            .where(WorkerCameraAssignment.released_at.is_(None)))).all()
        for a, cam, w in rows:
            stop = cam.desired_state == DesiredState.STOPPED
            outdated = a.camera_generation < cam.config_generation
            if not (stop or outdated) or a.state == AssignmentState.RELEASING:
                continue
            a.state = AssignmentState.RELEASING
            a.reason = "stop requested" if stop else "configuration changed"
            self._camera_status(cam, CameraStatus.STOPPING, a.reason)
            self._command(w.node_id, PipelineCommandType.REMOVE_SOURCE, {"worker_id": w.id, "camera_id": cam.id})
            stats["sources_removing"] += 1
        # stopped cameras without assignment
        cams = (await db.execute(select(Camera).where(and_(
            Camera.desired_state == DesiredState.STOPPED,
            Camera.status.in_([CameraStatus.PENDING_CAPACITY, CameraStatus.SCHEDULED, CameraStatus.FAILED,
                               CameraStatus.STARTING, CameraStatus.RUNNING, CameraStatus.RECONNECTING]))))).scalars()
        for cam in cams:
            has = (await db.execute(select(func.count()).select_from(WorkerCameraAssignment).where(and_(
                WorkerCameraAssignment.camera_id == cam.id, WorkerCameraAssignment.released_at.is_(None))))).scalar()
            if not has:
                self._camera_status(cam, CameraStatus.STOPPED, None)
                if self.streams is not None:
                    await self.streams.remove_raw_path(cam.id)

    def _retry_due(self, cam: Camera) -> bool:
        if cam.status != CameraStatus.FAILED:
            return True
        changed = _aware(cam.status_changed_at)
        if changed is None:
            return True
        backoff = min(30 * 2 ** min(cam.rtsp_failures, 5), 600)
        return (utcnow() - changed).total_seconds() >= backoff

    async def _start_requested(self, db: AsyncSession, stats) -> None:
        assigned = {a.camera_id for a in await self._active_assignments(db)}
        candidates = [c for c in (await db.execute(
            select(Camera).where(Camera.desired_state == DesiredState.RUNNING).order_by(Camera.created_at, Camera.id)
        )).scalars() if c.id not in assigned and self._retry_due(c)]
        if not candidates:
            return
        models, versions = await self._load_routing_catalog(db)
        amap = await self._assignments_for(db, [c.id for c in candidates])
        workers = {w.id: w for w in (await db.execute(select(Worker))).scalars()}
        active = await self._active_assignments(db)
        occupancy = defaultdict(int)
        for a in active:
            occupancy[a.worker_id] += 1
        gpus = await self._gpu_snapshots(db, active, workers)
        gpu_rows = {g.id: g for g in (await db.execute(select(GpuDevice))).scalars()}

        for cam in candidates:
            try:
                plan = build_plan(cam, amap.get(cam.id, []), models, versions, self.s.cameras_per_worker)
            except RoutingError as e:
                self._camera_status(cam, CameraStatus.FAILED, f"routing: {e}")
                stats["routing_failed"] += 1
                continue

            # ----- admission control across GPUs
            best: tuple[float, GpuSnapshot] | None = None
            reasons = []
            for g in gpus:
                cost = await self._stream_cost(db, g.name, plan, versions)
                d = admit(g, cost, self.s)
                if d.admitted:
                    # prefer a GPU that already has a worker with a free slot for this signature
                    bonus = 1.0 if any(w.gpu_device_id == g.device_id and w.pipeline_signature == plan.signature
                                       and w.state in LIVE_WORKER_STATES and occupancy[w.id] < w.capacity
                                       for w in workers.values()) else 0.0
                    score = d.headroom + bonus
                    if best is None or score > best[0]:
                        best = (score, g)
                else:
                    reasons.append(f"gpu{g.device_id}: {d.reason}")
            if best is None:
                reason = "; ".join(reasons) if reasons else "no GPU registered (is the node agent running?)"
                self._camera_status(cam, CameraStatus.PENDING_CAPACITY, reason[:1000])
                stats["pending_capacity"] += 1
                continue
            gpu = best[1]
            gdev = gpu_rows[gpu.device_id]

            # ----- find or create worker
            worker = next((w for w in sorted(workers.values(), key=lambda w: w.created_at or utcnow())
                           if w.pipeline_signature == plan.signature and w.gpu_device_id == gpu.device_id
                           and w.state in LIVE_WORKER_STATES and w.desired_state == DesiredState.RUNNING
                           and occupancy[w.id] < w.capacity), None)
            src = self.build_source_spec(cam)
            if worker is None:
                worker = Worker(
                    id=f"WORKER_{uuid.uuid4().hex[:10].upper()}", node_id=gdev.node_id, gpu_device_id=gdev.id,
                    pipeline_signature=plan.signature, model_set=plan.model_set, capacity=self.s.cameras_per_worker,
                    state=WorkerState.STARTING, desired_state=DesiredState.RUNNING, state_changed_at=utcnow(),
                    created_at=utcnow(),
                )
                db.add(worker)
                workers[worker.id] = worker
                await db.flush()
                spec = self.build_worker_spec(worker, plan, gdev.gpu_index, [src])
                self._command(gdev.node_id, PipelineCommandType.START_WORKER, {"worker_id": worker.id, "spec": spec})
                stats["workers_created"] += 1
                SCHED.workers_created.inc()
            else:
                self._command(worker.node_id, PipelineCommandType.ADD_SOURCE,
                              {"worker_id": worker.id, "source": src, "plan_signature": plan.signature})
                stats["sources_added"] += 1
            db.add(WorkerCameraAssignment(worker_id=worker.id, camera_id=cam.id, camera_generation=cam.config_generation,
                                          state=AssignmentState.ASSIGNING, reason="scheduled"))
            occupancy[worker.id] += 1
            gpu.active_streams += 1
            gpu.unconfirmed_streams += 1
            self._camera_status(cam, CameraStatus.STARTING, f"assigned to {worker.id} on gpu {gdev.gpu_index}")
            if self.streams is not None and cam.stream_mode in (StreamMode.RAW, StreamMode.BOTH):
                await self.streams.ensure_raw_path(cam)
            stats["assigned"] += 1

    async def _retire_empty_workers(self, db: AsyncSession, stats) -> None:
        counts = dict((await db.execute(
            select(WorkerCameraAssignment.worker_id, func.count())
            .where(WorkerCameraAssignment.released_at.is_(None)).group_by(WorkerCameraAssignment.worker_id))).all())
        for w in (await db.execute(select(Worker).where(Worker.state.in_(LIVE_WORKER_STATES)))).scalars():
            if counts.get(w.id, 0) == 0 or w.desired_state == DesiredState.STOPPED:
                w.desired_state = DesiredState.STOPPED
                w.state = WorkerState.STOPPING
                w.state_changed_at = utcnow()
                self._command(w.node_id, PipelineCommandType.STOP_WORKER, {"worker_id": w.id, "reason": "no cameras assigned"})
                stats["workers_stopping"] += 1

    async def _update_gauges(self) -> None:
        async with self.sm() as db:
            for st, n in (await db.execute(select(Camera.status, func.count()).group_by(Camera.status))).all():
                SCHED.cameras.labels(st.value if hasattr(st, "value") else st).set(n)
            for st, n in (await db.execute(select(Worker.state, func.count()).group_by(Worker.state))).all():
                SCHED.workers.labels(st.value if hasattr(st, "value") else st).set(n)

    # =========================================================== API actions
    async def request_worker_restart(self, db: AsyncSession, worker_id: str) -> Worker:
        """Restart a worker in place: only its own cameras are interrupted."""
        w = await db.get(Worker, worker_id)
        if w is None:
            raise KeyError(worker_id)
        rows = (await db.execute(select(WorkerCameraAssignment, Camera)
                                 .join(Camera, Camera.id == WorkerCameraAssignment.camera_id)
                                 .where(and_(WorkerCameraAssignment.worker_id == w.id,
                                             WorkerCameraAssignment.released_at.is_(None))))).all()
        if not rows:
            raise ValueError("worker has no cameras; nothing to restart")
        models, versions = await self._load_routing_catalog(db)
        amap = await self._assignments_for(db, [c.id for _, c in rows])
        plan = build_plan(rows[0][1], amap[rows[0][1].id], models, versions, self.s.cameras_per_worker)
        gdev = await db.get(GpuDevice, w.gpu_device_id) if w.gpu_device_id else None
        w.spec_generation += 1
        w.restart_count += 1
        w.state = WorkerState.STARTING
        w.state_changed_at = utcnow()
        w.last_heartbeat_at = None
        for a, _ in rows:
            a.state = AssignmentState.ASSIGNING
        spec = self.build_worker_spec(w, plan, gdev.gpu_index if gdev else 0, [self.build_source_spec(c) for _, c in rows])
        self._command(w.node_id, PipelineCommandType.RESTART_WORKER, {"worker_id": w.id, "spec": spec})
        return w

    async def request_worker_stop(self, db: AsyncSession, worker_id: str) -> Worker:
        """Stop a worker; its cameras (if still desired RUNNING) get rescheduled elsewhere."""
        w = await db.get(Worker, worker_id)
        if w is None:
            raise KeyError(worker_id)
        w.desired_state = DesiredState.STOPPED
        w.state = WorkerState.STOPPING
        w.state_changed_at = utcnow()
        self._command(w.node_id, PipelineCommandType.STOP_WORKER, {"worker_id": w.id, "reason": "api stop"})
        return w

    # =========================================================== event handling
    async def _seen(self, db: AsyncSession, event_id: str | None) -> bool:
        if not event_id:
            return False
        if await db.get(ProcessedCommand, event_id) is not None:
            return True
        db.add(ProcessedCommand(command_id=event_id, source=self.source))
        return False

    async def handle_message(self, topic: str, key: str | None, msg: dict[str, Any]) -> None:
        async with self.sm() as db:
            try:
                if topic == Topics.WORKER_HEARTBEAT:
                    await self._on_heartbeat(db, msg)
                elif topic == Topics.PIPELINE_EVENTS:
                    if await self._seen(db, msg.get("event_id")):
                        return
                    await self._on_pipeline_event(db, msg)
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def _on_heartbeat(self, db: AsyncSession, hb: dict[str, Any]) -> None:
        now = utcnow()
        if hb.get("kind") == "node":
            for g in hb.get("gpus", []):
                dev = (await db.execute(select(GpuDevice).where(GpuDevice.uuid == g["uuid"]))).scalar_one_or_none()
                if dev is None:
                    dev = GpuDevice(node_id=hb["node_id"], gpu_index=g["index"], uuid=g["uuid"], name=g.get("name"))
                    db.add(dev)
                dev.node_id = hb["node_id"]
                dev.gpu_index = g["index"]
                dev.name = g.get("name")
                dev.memory_total_mb = int(g.get("memory_total_mb") or 0)
                dev.driver_version = g.get("driver_version")
                dev.latest_metrics = g
                dev.last_seen_at = now
                temp = g.get("temperature_c")
                dev.healthy = not (temp is not None and temp >= self.s.gpu_max_temperature_c + 5)
                await db.flush()
                last = self._last_gpu_sample.get(dev.id)
                if last is None or (now - last).total_seconds() >= 30:
                    self._last_gpu_sample[dev.id] = now
                    db.add(GpuResourceMetric(
                        gpu_device_id=dev.id, ts=now, gpu_util_pct=g.get("gpu_util_pct"),
                        memory_used_mb=g.get("memory_used_mb"), memory_total_mb=g.get("memory_total_mb"),
                        decoder_util_pct=g.get("decoder_util_pct"), encoder_util_pct=g.get("encoder_util_pct"),
                        temperature_c=g.get("temperature_c"), power_w=g.get("power_w")))
            return

        wid = hb.get("worker_id")
        w = await db.get(Worker, wid) if wid else None
        if w is None:
            return
        if w.state in (WorkerState.FAILED, WorkerState.STOPPED):
            # a worker we already gave up on is still alive: tear it down
            self._command(w.node_id, PipelineCommandType.STOP_WORKER, {"worker_id": w.id, "reason": "stale worker"})
            return
        if hb.get("spec_generation") is not None and hb["spec_generation"] < w.spec_generation:
            return  # heartbeat from previous incarnation during a restart
        w.last_heartbeat_at = now
        w.pid = hb.get("pid")
        w.metrics = hb.get("metrics") or {}
        reported = hb.get("state")
        if w.state != WorkerState.STOPPING and reported in ("RUNNING", "DEGRADED"):
            new = WorkerState(reported)
            if new != w.state:
                w.state = new
                w.state_changed_at = now
                w.started_at = w.started_at or now
            w.health = "healthy" if new == WorkerState.RUNNING else "degraded"
        for cam_id, info in (hb.get("sources") or {}).items():
            await self._apply_source_status(db, w, cam_id, info.get("status"), info.get("reason"))

    async def _apply_source_status(self, db: AsyncSession, w: Worker, cam_id: str, status: str | None, reason: str | None) -> None:
        a = (await db.execute(select(WorkerCameraAssignment).where(and_(
            WorkerCameraAssignment.camera_id == cam_id, WorkerCameraAssignment.worker_id == w.id,
            WorkerCameraAssignment.released_at.is_(None))))).scalar_one_or_none()
        cam = await db.get(Camera, cam_id)
        if a is None or cam is None or a.state == AssignmentState.RELEASING:
            return
        if status == "RUNNING":
            if a.state != AssignmentState.ACTIVE:
                a.state = AssignmentState.ACTIVE
                db.add(PipelineSession(worker_id=w.id, camera_id=cam_id))
            if cam.status != CameraStatus.RUNNING:
                cam.rtsp_failures = 0
            self._camera_status(cam, CameraStatus.RUNNING, None)
        elif status == "RECONNECTING":
            if cam.status != CameraStatus.RECONNECTING:
                cam.rtsp_failures += 1
                PIPE.rtsp_failures.labels(cam_id).inc()
            self._camera_status(cam, CameraStatus.RECONNECTING, reason or "rtsp reconnecting")

    async def _on_pipeline_event(self, db: AsyncSession, env: dict[str, Any]) -> None:
        et = env.get("event_type")
        p = env.get("payload", {})
        w = await db.get(Worker, p.get("worker_id")) if p.get("worker_id") else None
        if w is None:
            return
        now = utcnow()
        if et == PipelineEventType.WORKER_RUNNING.value and w.state in (WorkerState.STARTING, WorkerState.CREATED, WorkerState.DEGRADED):
            w.state, w.state_changed_at, w.started_at, w.health = WorkerState.RUNNING, now, w.started_at or now, "healthy"
            w.last_heartbeat_at = now
        elif et == PipelineEventType.WORKER_DEGRADED.value and w.state == WorkerState.RUNNING:
            w.state, w.state_changed_at, w.health = WorkerState.DEGRADED, now, "degraded"
            w.last_error = p.get("reason")
        elif et == PipelineEventType.WORKER_STOPPED.value:
            await self._finalize_worker(db, w, WorkerState.STOPPED, p.get("reason") or "stopped")
        elif et == PipelineEventType.WORKER_FAILED.value:
            await self._finalize_worker(db, w, WorkerState.FAILED, p.get("reason") or "failed")
            SCHED.worker_failures.inc()
        elif et == PipelineEventType.MODEL_LOAD_FAILED.value:
            await self._on_model_load_failed(db, w, p)
        elif et in (PipelineEventType.SOURCE_RUNNING.value, PipelineEventType.SOURCE_RECONNECTING.value):
            await self._apply_source_status(db, w, p["camera_id"], "RUNNING" if et.endswith("RUNNING") else "RECONNECTING", p.get("reason"))
        elif et in (PipelineEventType.SOURCE_REMOVED.value, PipelineEventType.SOURCE_FAILED.value):
            cam_id = p.get("camera_id")
            a = (await db.execute(select(WorkerCameraAssignment).where(and_(
                WorkerCameraAssignment.camera_id == cam_id, WorkerCameraAssignment.worker_id == w.id,
                WorkerCameraAssignment.released_at.is_(None))))).scalar_one_or_none()
            if a is not None:
                a.state, a.released_at = AssignmentState.RELEASED, now
                a.reason = (a.reason or et)[:255]
            await self._end_pipeline_session(db, cam_id, w.id, et)
            cam = await db.get(Camera, cam_id)
            if cam is not None:
                if et == PipelineEventType.SOURCE_FAILED.value:
                    cam.rtsp_failures += 1
                    self._camera_status(cam, CameraStatus.FAILED, p.get("reason") or "source failed")
                elif cam.desired_state == DesiredState.STOPPED:
                    self._camera_status(cam, CameraStatus.STOPPED, None)
                else:
                    self._camera_status(cam, CameraStatus.SCHEDULED, "re-scheduling after configuration change")
        elif et == PipelineEventType.COMMAND_REJECTED.value:
            log.error("command rejected by worker: %s", p.get("reason"), extra={"worker_id": w.id})

    async def _on_model_load_failed(self, db: AsyncSession, w: Worker, p: dict[str, Any]) -> None:
        """Mark the version FAILED and roll back to the last good version (if any)."""
        model_id, vid = p.get("model_id"), p.get("model_version_id")
        model = await db.get(Model, model_id) if model_id else None
        if model_id:
            PIPE.model_load_failures.labels(model_id).inc()
        if model is not None and vid is not None:
            v = await db.get(ModelVersion, vid)
            if v is not None:
                v.status = ModelVersionStatus.FAILED
                v.failure_reason = (p.get("reason") or "engine load failed")[:2000]
            if model.active_version_id == vid:
                prev = (await db.execute(select(ModelVersion).where(and_(
                    ModelVersion.model_id == model.id, ModelVersion.id != vid,
                    ModelVersion.status == ModelVersionStatus.RETIRED,
                    ModelVersion.activated_at.is_not(None))).order_by(ModelVersion.activated_at.desc()).limit(1))).scalar_one_or_none()
                if prev is not None:
                    prev.status = ModelVersionStatus.ACTIVE
                    prev.activated_at = utcnow()
                    model.active_version_id = prev.id
                    log.warning("rolled back model %s to version %s", model.id, prev.version, extra={"model_id": model.id})
                else:
                    model.active_version_id = None
                await bump_cameras_for_model(db, model.id)
        await self._finalize_worker(db, w, WorkerState.FAILED, f"model load failed: {model_id}")


async def bump_cameras_for_model(db: AsyncSession, model_id: str) -> int:
    """Increment config_generation of cameras using a model so their pipelines are rebuilt."""
    cam_ids = [r for (r,) in (await db.execute(
        select(CameraModelAssignment.camera_id).where(CameraModelAssignment.model_id == model_id))).all()]
    if cam_ids:
        await db.execute(update(Camera).where(Camera.id.in_(cam_ids)).values(
            config_generation=Camera.config_generation + 1))
    return len(cam_ids)
