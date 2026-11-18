from datetime import datetime, timezone

from app.bus.schemas import Envelope, Topics
from app.db.models import (
    Camera,
    CameraModelAssignment,
    DesiredState,
    ExecutionStrategy,
    InferenceBackend,
    Model,
    ModelTask,
    ModelVersion,
    ModelVersionStatus,
)

# Test catalogue standing in for the three project models (names are placeholders;
# register the real ones through POST /api/v1/models).
MODELS = [("model_a", ModelTask.DETECTOR), ("model_b", ModelTask.DETECTOR), ("model_c", ModelTask.POSE),
          ("reid", ModelTask.REID)]


async def seed_models(db):
    for mid, task in MODELS:
        m = Model(id=mid, name=mid, task=task, backend=InferenceBackend.NVINFER)
        db.add(m)
        await db.flush()
        v = ModelVersion(model_id=mid, version="1", status=ModelVersionStatus.ACTIVE, onnx_file=f"{mid}/model.onnx",
                         labels=["person"], activated_at=datetime.now(timezone.utc),
                         embedding_dim=256 if task == ModelTask.REID else None)
        db.add(v)
        await db.flush()
        m.active_version_id = v.id
    await db.commit()


async def add_cameras(db, ids, models=("model_a",), strategy=ExecutionStrategy.INDEPENDENT, parents=None, running=True):
    for cid in ids:
        db.add(Camera(id=cid, name=cid, rtsp_url=f"rtsp://10.0.0.1/{cid}", execution_strategy=strategy,
                      desired_state=DesiredState.RUNNING if running else DesiredState.STOPPED))
        for i, mid in enumerate(models):
            db.add(CameraModelAssignment(camera_id=cid, model_id=mid, order=i, parent_model_id=(parents or {}).get(mid)))
    await db.commit()


def commands(bus, ctype=None):
    out = [p for p in bus.on(Topics.PIPELINE_COMMANDS)]
    return [c for c in out if ctype is None or c["event_type"] == ctype]


def pipeline_event(etype, **payload):
    return Envelope(event_type=etype, source="test", payload=payload).model_dump()


def worker_heartbeat(worker_id, sources, state="RUNNING", gen=1):
    return {"kind": "worker", "node_id": "gpu-node-1", "worker_id": worker_id, "state": state, "pid": 1234,
            "spec_generation": gen, "sources": {c: {"status": s} for c, s in sources.items()},
            "metrics": {"avg_latency_ms": 45.0}}
