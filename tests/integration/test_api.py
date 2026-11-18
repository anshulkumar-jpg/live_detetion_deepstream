"""End-to-end API tests (FastAPI app + real PostgreSQL; MediaMTX API mocked at HTTP level)."""
import time

import httpx
import pytest
from sqlalchemy import func, select

from app.bus.schemas import Topics
from app.db.models import AuditLog, Camera, CameraStatus
from app.main import create_app
from app.services.mediamtx import MediaMTXClient, StreamManager
from tests.conftest import gpu_heartbeat
from tests.integration.helpers import worker_heartbeat

pytestmark = pytest.mark.integration


def mediamtx_mock():
    calls = []

    def handler(req: httpx.Request):
        calls.append((req.method, req.url.path))
        if req.url.path.startswith("/v3/paths/get/cam/CAM_1/processed"):
            return httpx.Response(200, json={"name": "cam/CAM_1/processed", "ready": True,
                                             "readers": [{"type": "webRTCSession"}], "tracks": ["H264"]})
        if req.url.path.startswith("/v3/paths/get/"):
            return httpx.Response(404, json={"error": "path not found"})
        return httpx.Response(200, json={})
    return httpx.MockTransport(handler), calls


@pytest.fixture
async def api(engine, settings, bus):
    transport, calls = mediamtx_mock()
    streams = StreamManager(settings, MediaMTXClient(settings, client=httpx.AsyncClient(
        transport=transport, base_url="http://mediamtx:9997")))
    app = create_app(settings, bus=bus, streams=streams)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/v1/auth/token", json={"username": "admin", "password": "admin-password-123"})
            assert r.status_code == 200
            c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
            c.app = app
            c.mtx_calls = calls
            yield c


MODEL = {"id": "model_a", "name": "Model A", "task": "detector",
         "version": {"version": "1", "onnx_file": "model_a/model.onnx", "labels": ["person", "car"],
                     "nvinfer_properties": {"net-scale-factor": 0.0039215697, "maintain-aspect-ratio": 1}}}


async def setup_models(c):
    assert (await c.post("/api/v1/models", json=MODEL)).status_code == 201
    pose = {"id": "model_c", "name": "Pose", "task": "pose",
            "version": {"version": "1", "onnx_file": "model_c/pose.onnx"}}
    assert (await c.post("/api/v1/models", json=pose)).status_code == 201
    vehicle = {"id": "model_b", "name": "Vehicle", "task": "detector",
               "version": {"version": "1", "engine_file": "model_b/model_b.engine", "labels": ["car", "truck"]}}
    assert (await c.post("/api/v1/models", json=vehicle)).status_code == 201


async def test_auth_and_rbac(api):
    anon = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test")
    assert (await anon.get("/api/v1/cameras")).status_code == 401
    assert (await anon.post("/api/v1/auth/token", json={"username": "admin", "password": "nope"})).status_code == 401
    assert (await api.post("/api/v1/auth/users", json={"username": "viewer1", "password": "viewer-password-1",
                                                       "role": "viewer"})).status_code == 201
    tok = (await anon.post("/api/v1/auth/token", json={"username": "viewer1", "password": "viewer-password-1"})).json()
    anon.headers["Authorization"] = f"Bearer {tok['access_token']}"
    assert (await anon.get("/api/v1/cameras")).status_code == 200
    r = await anon.post("/api/v1/cameras", json={"id": "X", "name": "x", "rtsp_url": "rtsp://h/x"})
    assert r.status_code == 403
    internal = httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test",
                                 headers={"x-internal-token": "internal-test-token"})
    assert (await internal.get("/api/v1/models")).status_code == 200


async def test_camera_crud_hides_credentials(api, sm):
    r = await api.post("/api/v1/cameras", json={"id": "CAM_1", "name": "Gate", "rtsp_url": "rtsp://admin:S3cr3t!@10.1.1.5:554/live",
                                                "stream_mode": "annotated"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["rtsp_url"] == "rtsp://10.1.1.5:554/live" and body["has_credentials"] is True
    assert "S3cr3t" not in r.text
    assert (await api.post("/api/v1/cameras", json={"id": "CAM_1", "name": "dup", "rtsp_url": "rtsp://h/x"})).status_code == 409
    assert (await api.post("/api/v1/cameras", json={"id": "bad id!", "name": "x", "rtsp_url": "rtsp://h"})).status_code == 422
    assert (await api.post("/api/v1/cameras", json={"id": "C2", "name": "x", "rtsp_url": "http://h"})).status_code == 422
    async with sm() as db:
        cam = await db.get(Camera, "CAM_1")
        assert "S3cr3t" not in (cam.password_enc or "") and cam.password_enc
    r = await api.put("/api/v1/cameras/CAM_1", json={"name": "Main gate"})
    assert r.json()["config_generation"] == 1  # metadata change does not rebuild pipeline
    r = await api.put("/api/v1/cameras/CAM_1", json={"password": "new"})
    assert r.json()["config_generation"] == 2
    assert (await api.get("/api/v1/cameras/NOPE")).status_code == 404
    assert (await api.delete("/api/v1/cameras/CAM_1")).status_code == 202
    async with sm() as db:
        assert (await db.execute(select(func.count()).select_from(AuditLog))).scalar() >= 4


async def test_models_assignment_modes_and_versions(api):
    await setup_models(api)
    r = await api.get("/api/v1/models")
    assert {m["id"] for m in r.json()} == {"model_a", "model_b", "model_c"}
    assert r.json()[0]["active_version"]["version"] == "1"
    for cid in ("C101", "C104", "C106"):
        await api.post("/api/v1/cameras", json={"id": cid, "name": cid, "rtsp_url": f"rtsp://h/{cid}"})
    # Mode A
    r = await api.post("/api/v1/cameras/C101/models", json={"models": [{"model_id": "model_a"}]})
    assert r.status_code == 200 and [m["model_id"] for m in r.json()["models"]] == ["model_a"]
    # Mode B
    r = await api.post("/api/v1/cameras/C104/models", json={"models": [
        {"model_id": "model_a"}, {"model_id": "model_c"}, {"model_id": "model_b", "enabled": False}], "tracking_enabled": True})
    assert r.status_code == 200 and len(r.json()["models"]) == 3
    # Mode C (cascade inferred from parent_model_id)
    r = await api.post("/api/v1/cameras/C106/models", json={"models": [
        {"model_id": "model_a"}, {"model_id": "model_c", "parent_model_id": "model_a", "operate_on_class_ids": [0]}]})
    assert r.status_code == 200 and r.json()["execution_strategy"] == "cascade"
    bad = await api.post("/api/v1/cameras/C101/models", json={"models": [{"model_id": "ghost"}]})
    assert bad.status_code == 422
    bad = await api.post("/api/v1/cameras/C101/models", json={"models": [
        {"model_id": "model_c", "parent_model_id": "model_b"}]})
    assert bad.status_code == 422  # parent not assigned
    # new version + rollback
    r = await api.put("/api/v1/models/model_a", json={"new_version": {"version": "2", "onnx_file": "model_a/v2.onnx"}})
    assert r.json()["active_version"]["version"] == "2"
    r = await api.put("/api/v1/models/model_a", json={"activate_version": "1"})
    assert r.json()["active_version"]["version"] == "1"
    assert {v["status"] for v in r.json()["versions"]} == {"ACTIVE", "RETIRED"}
    cam = (await api.get("/api/v1/cameras/C101")).json()
    assert cam["config_generation"] >= 3  # each activation rebuilt cameras using model_a
    bad = await api.post("/api/v1/models", json={"id": "int8", "name": "x", "task": "detector",
                                                 "version": {"version": "1", "onnx_file": "a.onnx", "precision": "int8"}})
    assert bad.status_code == 422
    bad = await api.post("/api/v1/models", json={"id": "trav", "name": "x", "task": "detector",
                                                 "version": {"version": "1", "onnx_file": "../../etc/passwd"}})
    assert bad.status_code == 422


async def test_start_stop_end_to_end_with_scheduler(api, sm, bus):
    await setup_models(api)
    await api.post("/api/v1/cameras", json={"id": "CAM_1", "name": "a", "rtsp_url": "rtsp://u:p@h/1", "stream_mode": "both"})
    assert (await api.post("/api/v1/cameras/CAM_1/start")).status_code == 422  # no models yet
    await api.post("/api/v1/cameras/CAM_1/models", json={"models": [{"model_id": "model_a"}]})
    r = await api.post("/api/v1/cameras/CAM_1/start")
    assert r.status_code == 202 and r.json()["desired_state"] == "RUNNING"
    sched = api.app.state.scheduler
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat())
    await sched.reconcile_once()
    cam = (await api.get("/api/v1/cameras/CAM_1")).json()
    assert cam["status"] == "STARTING" and cam["worker_id"]
    assert ("POST", "/v3/config/paths/add/cam/CAM_1/raw") in api.mtx_calls
    wid = cam["worker_id"]
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, worker_heartbeat(wid, {"CAM_1": "RUNNING"}))
    assert (await api.get("/api/v1/cameras/CAM_1")).json()["status"] == "RUNNING"
    ws = (await api.get("/api/v1/workers")).json()
    assert ws[0]["id"] == wid and ws[0]["camera_ids"] == ["CAM_1"] and ws[0]["state"] == "RUNNING"
    assert (await api.get(f"/api/v1/workers/{wid}/metrics")).json()["metrics"]["avg_latency_ms"] == 45.0
    st = (await api.get("/api/v1/cameras/CAM_1/stream")).json()
    assert st["streams"]["processed"]["hls"].endswith("/cam/CAM_1/processed/index.m3u8")
    status = (await api.get("/api/v1/cameras/CAM_1/stream/status")).json()
    proc = next(s for s in status["streams"] if s["kind"] == "processed")
    assert proc["ready"] is True and proc["viewers"] == 1
    res = (await api.get("/api/v1/gpu/resources")).json()
    assert res["gpus"][0]["active_streams"] == 1 and res["gpus"][0]["accepting_new_streams"]
    assert (await api.get("/api/v1/gpu/status")).json()["count"] == 1
    assert (await api.get("/api/v1/gpu/metrics")).status_code == 200
    r = await api.post(f"/api/v1/workers/{wid}/restart")
    assert r.status_code == 202 and r.json()["spec_generation"] == 2
    assert bus.on(Topics.PIPELINE_COMMANDS)[-1]["event_type"] == "RESTART_WORKER"
    r = await api.post("/api/v1/cameras/CAM_1/stop")
    assert r.json()["desired_state"] == "STOPPED"
    await sched.reconcile_once()
    assert bus.on(Topics.PIPELINE_COMMANDS)[-1]["event_type"] == "REMOVE_SOURCE"
    assert (await api.delete("/api/v1/cameras/CAM_1")).status_code == 409  # still attached


async def test_system_and_analytics_endpoints(api):
    assert (await api.get("/health")).json() == {"status": "ok"}
    r = await api.get("/ready")
    assert r.status_code == 200 and r.json()["checks"]["database"]
    m = await api.get("/metrics")
    assert "vms_api_requests_total" in m.text
    assert (await api.get("/api/v1/events")).json()["total"] == 0
    s = (await api.get("/api/v1/analytics/summary")).json()
    assert s["window_hours"] == 24
    assert (await api.get("/api/v1/analytics/cameras/NOPE")).status_code == 404
    assert "x-request-id" in (await api.get("/health")).headers


async def test_register_and_schedule_1000_cameras(api, sm, bus):
    await setup_models(api)
    t0 = time.perf_counter()
    for chunk in range(10):
        cams = [{"id": f"CAM_{chunk * 100 + i:04d}", "name": f"cam {chunk * 100 + i}",
                 "rtsp_url": f"rtsp://user:pass@10.0.{chunk}.{i}/stream1"} for i in range(100)]
        r = await api.post("/api/v1/cameras/bulk", json={"cameras": cams})
        assert r.status_code == 201
    reg_s = time.perf_counter() - t0
    page = (await api.get("/api/v1/cameras", params={"limit": 1000})).json()
    assert page["total"] == 1000 and len(page["items"]) == 1000
    # assign + start all (desired state), then schedule against a benchmarked capacity of 40 streams
    from app.db.models import CameraModelAssignment, DesiredState
    async with sm() as db:
        for i in range(1000):
            db.add(CameraModelAssignment(camera_id=f"CAM_{i:04d}", model_id="model_a", order=0))
        from sqlalchemy import update
        await db.execute(update(Camera).values(desired_state=DesiredState.RUNNING, status=CameraStatus.SCHEDULED))
        await db.commit()
    sched = api.app.state.scheduler
    await sched.handle_message(Topics.WORKER_HEARTBEAT, None, gpu_heartbeat(mem_total=81920, mem_used=1000))
    await sched.reconcile_once()  # learn signature
    async with sm() as db:
        from app.db.models import Worker
        sig = (await db.execute(select(Worker.pipeline_signature))).scalars().first()
    r = await api.post("/api/v1/gpu/capacity-profiles", json={"gpu_name": "NVIDIA L4", "pipeline_signature": sig,
                                                              "max_streams": 40, "memory_mb_per_stream": 300})
    assert r.status_code == 201
    async with sm() as db:
        n_assigned = (await db.execute(select(func.count()).select_from(Camera).where(
            Camera.status == CameraStatus.STARTING))).scalar()
    t1 = time.perf_counter()
    await sched.reconcile_once()
    sched_s = time.perf_counter() - t1
    async with sm() as db:
        starting = (await db.execute(select(func.count()).select_from(Camera).where(
            Camera.status == CameraStatus.STARTING))).scalar()
        pending = (await db.execute(select(func.count()).select_from(Camera).where(
            Camera.status == CameraStatus.PENDING_CAPACITY))).scalar()
    # first pass (no profile) admitted by memory only; with the profile no further streams beyond 40 are admitted
    assert starting == max(n_assigned, 40) and starting + pending == 1000
    res = (await api.get("/api/v1/gpu/resources")).json()
    assert res["pending_capacity"]["count"] == pending
    print(f"\n1000-camera registration: {reg_s:.2f}s, steady-state reconcile pass: {sched_s:.2f}s, "
          f"admitted={starting}, pending={pending}")


async def test_mediamtx_auth_callback(api, settings):
    from app.core.security import Role, create_access_token
    tok = create_access_token("viewer", Role.VIEWER, settings)
    cb = "/internal/mediamtx/auth"
    ok = await api.post(cb, json={"action": "read", "path": "cam/CAM_1/processed", "token": tok, "ip": "8.8.8.8"})
    assert ok.status_code == 200
    q = await api.post(cb, json={"action": "read", "path": "cam/CAM_1/processed", "query": f"jwt={tok}", "ip": "8.8.8.8"})
    assert q.status_code == 200
    assert (await api.post(cb, json={"action": "read", "path": "cam/CAM_1/processed", "token": "bad"})).status_code == 401
    pub = {"action": "publish", "path": "cam/CAM_1/processed", "ip": "172.18.0.5", "user": "worker",
           "password": "internal-test-token"}
    assert (await api.post(cb, json=pub)).status_code == 200
    assert (await api.post(cb, json={**pub, "ip": "8.8.8.8"})).status_code == 401
    assert (await api.post(cb, json={**pub, "password": "x"})).status_code == 401
    assert (await api.post(cb, json={**pub, "path": "cam/CAM_1/raw"})).status_code == 401
    assert (await api.post(cb, json={"action": "api", "ip": "10.0.0.2"})).status_code == 200
    assert (await api.post(cb, json={"action": "api", "ip": "1.2.3.4"})).status_code == 401
