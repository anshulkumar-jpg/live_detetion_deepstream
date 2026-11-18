"""Scheduler / worker lifecycle / admission / recovery against real PostgreSQL."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, update

from app.bus.schemas import Topics
from app.db.models import (
    AssignmentState,
    Camera,
    CameraStatus,
    CapacityProfile,
    DesiredState,
    ExecutionStrategy,
    Model,
    ModelVersion,
    ModelVersionStatus,
    Worker,
    WorkerCameraAssignment,
    WorkerState,
)
from app.services.scheduler import Scheduler
from tests.conftest import gpu_heartbeat
from tests.integration.helpers import add_cameras, commands, pipeline_event, seed_models, worker_heartbeat

pytestmark = pytest.mark.integration


async def active_by_worker(db):
    rows = (await db.execute(select(WorkerCameraAssignment.worker_id, WorkerCameraAssignment.camera_id).where(
        WorkerCameraAssignment.released_at.is_(None)))).all()
    out = {}
    for w, c in rows:
        out.setdefault(w, set()).add(c)
    return out


@pytest.fixture
async def sched(sm, bus, settings):
    async with sm() as db:
        await seed_models(db)
    s = Scheduler(sm, bus, settings)
    await s.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat())
    return s


@pytest.mark.parametrize("n,expected_workers", [(4, 2), (6, 3), (10, 5), (5, 3)])
async def test_two_cameras_per_worker(sched, sm, bus, n, expected_workers):
    async with sm() as db:
        await add_cameras(db, [f"CAM_{i:03d}" for i in range(n)])
    stats = await sched.reconcile_once()
    assert stats["assigned"] == n and stats["workers_created"] == expected_workers
    async with sm() as db:
        groups = await active_by_worker(db)
        assert len(groups) == expected_workers
        assert all(len(c) <= 2 for c in groups.values())
        assert (await db.execute(select(func.count()).select_from(Camera).where(
            Camera.status == CameraStatus.STARTING))).scalar() == n
    starts = commands(bus, "START_WORKER")
    adds = commands(bus, "ADD_SOURCE")
    assert len(starts) == expected_workers and len(starts) + len(adds) == n
    spec = starts[0]["payload"]["spec"]
    assert spec["capacity"] == 2 and spec["plan"]["stages"][0]["model_id"] == "model_a"
    assert all(set(s) >= {"username_enc", "password_enc"} and "password" not in s for s in spec["sources"])


async def test_different_model_sets_never_share_a_worker(sched, sm):
    async with sm() as db:
        await add_cameras(db, ["A1", "A2", "A3"], models=("model_a",))
        await add_cameras(db, ["B1"], models=("model_a", "model_b"))
        await add_cameras(db, ["C1", "C2"], models=("model_a", "model_c"), strategy=ExecutionStrategy.CASCADE,
                          parents={"model_c": "model_a"})
    await sched.reconcile_once()
    async with sm() as db:
        groups = await active_by_worker(db)
        sigs = {w.id: w.pipeline_signature for w in (await db.execute(select(Worker))).scalars()}
    sets = sorted(sorted(c) for c in groups.values())
    assert sets == [["A1", "A2"], ["A3"], ["B1"], ["C1", "C2"]]
    by_sig = {}
    for wid, cams in groups.items():
        by_sig.setdefault(sigs[wid], set()).update(cams)
    assert len(by_sig) == 3


async def test_worker_lifecycle_add_remove_without_interrupting_others(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["CAM_1", "CAM_2"])
    await sched.reconcile_once()
    wid = commands(bus, "START_WORKER")[0]["payload"]["worker_id"]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, pipeline_event("WORKER_RUNNING", worker_id=wid))
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None,
                               worker_heartbeat(wid, {"CAM_1": "RUNNING", "CAM_2": "RUNNING"}))
    async with sm() as db:
        assert (await db.get(Worker, wid)).state == WorkerState.RUNNING
        assert {c.status for c in (await db.execute(select(Camera))).scalars()} == {CameraStatus.RUNNING}
        await db.execute(update(Camera).where(Camera.id == "CAM_1").values(desired_state=DesiredState.STOPPED))
        await db.commit()
    bus.messages.clear()
    await sched.reconcile_once()
    rm = commands(bus)
    assert [(c["event_type"], c["payload"].get("camera_id")) for c in rm] == [("REMOVE_SOURCE", "CAM_1")]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None,
                               pipeline_event("SOURCE_REMOVED", worker_id=wid, camera_id="CAM_1"))
    await sched.reconcile_once()
    async with sm() as db:
        assert (await db.get(Camera, "CAM_1")).status == CameraStatus.STOPPED
        assert (await db.get(Camera, "CAM_2")).status == CameraStatus.RUNNING
        assert (await db.get(Worker, wid)).state == WorkerState.RUNNING  # still serving CAM_2
        await db.execute(update(Camera).where(Camera.id == "CAM_2").values(desired_state=DesiredState.STOPPED))
        await db.commit()
    await sched.reconcile_once()
    await sched.handle_message(Topics.PIPELINE_EVENTS, None,
                               pipeline_event("SOURCE_REMOVED", worker_id=wid, camera_id="CAM_2"))
    bus.messages.clear()
    await sched.reconcile_once()  # worker now empty -> graceful shutdown
    assert [c["event_type"] for c in commands(bus)] == ["STOP_WORKER"]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, pipeline_event("WORKER_STOPPED", worker_id=wid))
    async with sm() as db:
        assert (await db.get(Worker, wid)).state == WorkerState.STOPPED


async def test_admission_control_pending_capacity_then_recovers(sm, bus, settings):
    async with sm() as db:
        await seed_models(db)
        await add_cameras(db, ["C1", "C2", "C3"])
    s = Scheduler(sm, bus, settings)
    # 8 GB GPU with 6 GB used and 1 GB reserve -> room for two 350 MB streams
    await s.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat(mem_total=8000, mem_used=6000))
    stats = await s.reconcile_once()
    assert stats["assigned"] == 2 and stats["pending_capacity"] == 1
    async with sm() as db:
        c3 = await db.get(Camera, "C3")
        assert c3.status == CameraStatus.PENDING_CAPACITY and "insufficient_gpu_memory" in c3.status_reason
    await s.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat(mem_total=8000, mem_used=3000))
    stats = await s.reconcile_once()
    assert stats["assigned"] == 1
    async with sm() as db:
        assert (await db.get(Camera, "C3")).status == CameraStatus.STARTING


async def test_no_gpu_registered_queues(sm, bus, settings):
    async with sm() as db:
        await seed_models(db)
        await add_cameras(db, ["C1"])
    s = Scheduler(sm, bus, settings)
    await s.reconcile_once()
    async with sm() as db:
        c = await db.get(Camera, "C1")
        assert c.status == CameraStatus.PENDING_CAPACITY and "no GPU" in c.status_reason
    assert not commands(bus)


async def test_overheated_or_stale_gpu_blocks(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["C1"])
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat(temp=92))
    await sched.reconcile_once()
    async with sm() as db:
        assert "gpu" in (await db.get(Camera, "C1")).status_reason


async def test_capacity_profile_limits_streams(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, [f"C{i}" for i in range(5)])
    # learn the signature, then record a benchmark saying this GPU handles 3 such streams
    await sched.reconcile_once()
    async with sm() as db:
        sig = (await db.execute(select(Worker.pipeline_signature))).scalars().first()
        await db.execute(update(Camera).values(desired_state=DesiredState.STOPPED))
        await db.commit()
    for _ in range(2):
        await sched.reconcile_once()
    async with sm() as db:
        for a in (await db.execute(select(WorkerCameraAssignment).where(WorkerCameraAssignment.released_at.is_(None)))).scalars():
            a.released_at, a.state = datetime.now(timezone.utc), AssignmentState.RELEASED
        await db.execute(update(Camera).values(desired_state=DesiredState.RUNNING, status=CameraStatus.SCHEDULED))
        db.add(CapacityProfile(gpu_name="NVIDIA L4", pipeline_signature=sig, max_streams=3, memory_mb_per_stream=300))
        await db.commit()
    stats = await sched.reconcile_once()
    assert stats["assigned"] == 3 and stats["pending_capacity"] == 2
    async with sm() as db:
        reasons = [c.status_reason for c in (await db.execute(select(Camera).where(
            Camera.status == CameraStatus.PENDING_CAPACITY))).scalars()]
        assert all("stream_capacity_reached" in r and "benchmark" in r for r in reasons)


async def test_worker_crash_recovery_leaves_healthy_workers_alone(sched, sm, bus, settings):
    async with sm() as db:
        await add_cameras(db, ["C1", "C2", "C3", "C4"])
    await sched.reconcile_once()
    w1, w2 = [c["payload"]["worker_id"] for c in commands(bus, "START_WORKER")]
    for w, cams in ((w1, ("C1", "C2")), (w2, ("C3", "C4"))):
        await sched.handle_message(Topics.WORKER_HEARTBEAT, None, worker_heartbeat(w, {c: "RUNNING" for c in cams}))
    async with sm() as db:  # w1 stops heartbeating
        await db.execute(update(Worker).where(Worker.id == w1).values(
            last_heartbeat_at=datetime.now(timezone.utc) - timedelta(seconds=settings.heartbeat_timeout_s + 5)))
        await db.commit()
    bus.messages.clear()
    stats = await sched.reconcile_once()
    assert stats["workers_failed"] == 1 and stats["workers_created"] == 1 and stats["assigned"] == 2
    cmds = commands(bus)
    assert all(c["payload"]["worker_id"] != w2 for c in cmds)  # healthy worker untouched
    stop = [c for c in cmds if c["event_type"] == "STOP_WORKER"]
    assert stop and stop[0]["payload"]["worker_id"] == w1
    new = [c for c in cmds if c["event_type"] == "START_WORKER"][0]
    assert sorted(s["camera_id"] for s in new["payload"]["spec"]["sources"]) == ["C1"]
    async with sm() as db:
        assert (await db.get(Worker, w1)).state == WorkerState.FAILED
        groups = await active_by_worker(db)
        assert groups[w2] == {"C3", "C4"}
        assert {"C1", "C2"} <= set().union(*[c for w, c in groups.items() if w != w2])


async def test_zombie_heartbeat_from_failed_worker_is_stopped(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["C1"])
    await sched.reconcile_once()
    wid = commands(bus, "START_WORKER")[0]["payload"]["worker_id"]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, pipeline_event("WORKER_FAILED", worker_id=wid, reason="x"))
    bus.messages.clear()
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, worker_heartbeat(wid, {"C1": "RUNNING"}))
    assert [c["event_type"] for c in commands(bus)] == ["STOP_WORKER"]


async def test_config_change_rebuilds_only_that_camera(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["C1", "C2"])
    await sched.reconcile_once()
    wid = commands(bus, "START_WORKER")[0]["payload"]["worker_id"]
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, worker_heartbeat(wid, {"C1": "RUNNING", "C2": "RUNNING"}))
    async with sm() as db:
        from app.db.models import CameraModelAssignment
        db.add(CameraModelAssignment(camera_id="C1", model_id="model_b", order=1))
        await db.execute(update(Camera).where(Camera.id == "C1").values(config_generation=Camera.config_generation + 1))
        await db.commit()
    bus.messages.clear()
    await sched.reconcile_once()
    assert [(c["event_type"], c["payload"].get("camera_id")) for c in commands(bus)] == [("REMOVE_SOURCE", "C1")]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, pipeline_event("SOURCE_REMOVED", worker_id=wid, camera_id="C1"))
    bus.messages.clear()
    await sched.reconcile_once()
    start = commands(bus, "START_WORKER")
    assert len(start) == 1 and [st["model_id"] for st in start[0]["payload"]["spec"]["plan"]["stages"]] == ["model_a", "model_b"]
    assert not [c for c in commands(bus) if c["payload"].get("worker_id") == wid]
    async with sm() as db:
        assert (await db.get(Camera, "C2")).status == CameraStatus.RUNNING


async def test_model_load_failure_rolls_back(sched, sm, bus):
    async with sm() as db:
        m = await db.get(Model, "model_a")
        v1 = await db.get(ModelVersion, m.active_version_id)
        v1.status = ModelVersionStatus.RETIRED
        v2 = ModelVersion(model_id="model_a", version="2", status=ModelVersionStatus.ACTIVE, onnx_file="a/v2.onnx",
                          activated_at=datetime.now(timezone.utc))
        db.add(v2)
        await db.flush()
        m.active_version_id = v2.id
        await db.commit()
        v1_id, v2_id = v1.id, v2.id
        await add_cameras(db, ["C1"])
    await sched.reconcile_once()
    start = commands(bus, "START_WORKER")[0]["payload"]
    assert start["spec"]["plan"]["stages"][0]["model_version"] == "2"
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, pipeline_event(
        "MODEL_LOAD_FAILED", worker_id=start["worker_id"], model_id="model_a", model_version_id=v2_id,
        reason="engine deserialization failed"))
    async with sm() as db:
        m = await db.get(Model, "model_a")
        assert m.active_version_id == v1_id
        assert (await db.get(ModelVersion, v2_id)).status == ModelVersionStatus.FAILED
        assert (await db.get(Worker, start["worker_id"])).state == WorkerState.FAILED
    bus.messages.clear()
    await sched.reconcile_once()
    restart = commands(bus, "START_WORKER")[0]["payload"]
    assert restart["spec"]["plan"]["stages"][0]["model_version"] == "1"


async def test_duplicate_events_are_idempotent(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["C1"])
    await sched.reconcile_once()
    wid = commands(bus, "START_WORKER")[0]["payload"]["worker_id"]
    ev = pipeline_event("SOURCE_FAILED", worker_id=wid, camera_id="C1", reason="401 Unauthorized")
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, ev)
    await sched.handle_message(Topics.PIPELINE_EVENTS, None, ev)
    async with sm() as db:
        c = await db.get(Camera, "C1")
        assert c.status == CameraStatus.FAILED and c.rtsp_failures == 1


async def test_failed_camera_retries_with_backoff(sched, sm, bus):
    async with sm() as db:
        await add_cameras(db, ["C1"])
    await sched.reconcile_once()
    wid = commands(bus, "START_WORKER")[0]["payload"]["worker_id"]
    await sched.handle_message(Topics.PIPELINE_EVENTS, None,
                               pipeline_event("SOURCE_FAILED", worker_id=wid, camera_id="C1", reason="timeout"))
    bus.messages.clear()
    await sched.reconcile_once()
    assert not commands(bus, "ADD_SOURCE") and not commands(bus, "START_WORKER")  # backoff not elapsed
    async with sm() as db:
        await db.execute(update(Camera).where(Camera.id == "C1").values(
            status_changed_at=datetime.now(timezone.utc) - timedelta(minutes=5)))
        await db.commit()
    await sched.reconcile_once()
    assert commands(bus, "ADD_SOURCE") or commands(bus, "START_WORKER")
