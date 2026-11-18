import pytest

from app.db.models import (
    Camera,
    CameraModelAssignment,
    ExecutionStrategy,
    InferenceBackend,
    Model,
    ModelTask,
    ModelVersion,
    ModelVersionStatus,
)
from app.services.gpu import GpuSnapshot, StreamCost, admit
from app.services.model_router import RoutingError, build_plan, validate_assignment_request
from worker.pipeline.events import UNTRACKED, EventEmitter, FrameResult, ObjectResult, SourceStats, TokenBucket


def catalog():
    models, versions = {}, {}
    for i, (mid, task) in enumerate([("person", ModelTask.DETECTOR), ("vehicle", ModelTask.DETECTOR),
                                     ("pose", ModelTask.POSE), ("reid", ModelTask.REID)], start=1):
        m = Model(id=mid, name=mid, task=task, backend=InferenceBackend.NVINFER, enabled=True, active_version_id=i)
        v = ModelVersion(id=i, model_id=mid, version="1", status=ModelVersionStatus.ACTIVE, onnx_file=f"{mid}.onnx",
                         labels=["person"] if mid == "person" else [], precision="fp16", max_batch_size=8,
                         nvinfer_properties={}, class_attrs={}, embedding_dim=256 if mid == "reid" else None)
        models[mid], versions[i] = m, v
    return models, versions


def cam(strategy=ExecutionStrategy.INDEPENDENT, tracking=True):
    return Camera(id="C1", name="c", rtsp_url="rtsp://x", tracking_enabled=tracking, tracker_type="NvDCF",
                  execution_strategy=strategy)


def a(mid, parent=None, enabled=True, order=0, classes=None):
    return CameraModelAssignment(camera_id="C1", model_id=mid, parent_model_id=parent, enabled=enabled, order=order,
                                 operate_on_class_ids=classes)


def test_mode_a_single_model():
    models, versions = catalog()
    plan = build_plan(cam(), [a("person")], models, versions, 2)
    assert [(s.model_id, s.process_mode, s.gie_unique_id, s.max_batch_size) for s in plan.stages] == [("person", 1, 1, 2)]


def test_mode_b_multi_model_and_disabled_ignored():
    models, versions = catalog()
    plan = build_plan(cam(), [a("person", order=0), a("vehicle", order=1), a("pose", order=2, enabled=False)],
                      models, versions, 2)
    assert [s.model_id for s in plan.primaries] == ["person", "vehicle"]
    assert not plan.secondaries


def test_mode_c_cascade():
    models, versions = catalog()
    plan = build_plan(cam(ExecutionStrategy.CASCADE),
                      [a("person"), a("pose", "person", order=1, classes=[0]), a("reid", "person", order=2)],
                      models, versions, 2)
    pose = next(s for s in plan.stages if s.model_id == "pose")
    reid = next(s for s in plan.stages if s.model_id == "reid")
    assert pose.process_mode == 2 and pose.operate_on_gie_id == 1 and pose.operate_on_class_ids == [0]
    assert reid.output_tensor_meta and reid.max_batch_size == 8 and reid.embedding_dim == 256


def test_signature_groups_identical_plans_only():
    models, versions = catalog()
    p1 = build_plan(cam(), [a("person")], models, versions, 2)
    p2 = build_plan(cam(), [a("person")], models, versions, 2)
    p3 = build_plan(cam(), [a("person"), a("vehicle", order=1)], models, versions, 2)
    p4 = build_plan(cam(tracking=False), [a("person")], models, versions, 2)
    assert p1.signature == p2.signature
    assert len({p1.signature, p3.signature, p4.signature}) == 3


def test_routing_errors():
    models, versions = catalog()
    with pytest.raises(RoutingError, match="no enabled"):
        build_plan(cam(), [a("person", enabled=False)], models, versions, 2)
    models["person"].enabled = False
    with pytest.raises(RoutingError, match="disabled"):
        build_plan(cam(), [a("person")], models, versions, 2)
    models["person"].enabled = True
    versions[1].status = ModelVersionStatus.FAILED
    with pytest.raises(RoutingError, match="FAILED"):
        build_plan(cam(), [a("person")], models, versions, 2)


def test_validate_request():
    models, _ = catalog()
    validate_assignment_request([{"model_id": "person"}, {"model_id": "pose"}], models, ExecutionStrategy.INDEPENDENT)
    with pytest.raises(RoutingError, match="cascade"):
        validate_assignment_request([{"model_id": "person"}, {"model_id": "pose", "parent_model_id": "person"}],
                                    models, ExecutionStrategy.INDEPENDENT)
    with pytest.raises(RoutingError, match="must cascade"):
        validate_assignment_request([{"model_id": "reid"}], models, ExecutionStrategy.INDEPENDENT)
    with pytest.raises(RoutingError, match="must be a detector"):
        validate_assignment_request([{"model_id": "pose"}, {"model_id": "reid", "parent_model_id": "pose"}],
                                    models, ExecutionStrategy.CASCADE)
    with pytest.raises(RoutingError, match="unknown"):
        validate_assignment_request([{"model_id": "nope"}], models, ExecutionStrategy.INDEPENDENT)


# ------------------------------------------------------------ admission
def snap(**m):
    base = {"memory_total_mb": 16000, "memory_used_mb": 4000, "gpu_util_pct": 30, "decoder_util_pct": 20,
            "encoder_util_pct": 0, "temperature_c": 60}
    base.update(m)
    return GpuSnapshot(device_id=1, name="L4", healthy=True, metrics=base, last_seen_age_s=2, active_streams=4)


def test_admit_ok(settings):
    d = admit(snap(), StreamCost(memory_mb=300), settings)
    assert d.admitted and 0 < d.headroom <= 1


@pytest.mark.parametrize("metrics,expect", [
    ({"memory_used_mb": 14800}, "insufficient_gpu_memory"),
    ({"gpu_util_pct": 95}, "gpu_utilization"),
    ({"decoder_util_pct": 90}, "nvdec_utilization"),
    ({"encoder_util_pct": 90}, "nvenc_utilization"),
    ({"temperature_c": 90}, "gpu_temperature"),
])
def test_admit_denials(settings, metrics, expect):
    d = admit(snap(**metrics), StreamCost(memory_mb=300), settings)
    assert not d.admitted and expect in d.reason


def test_admit_counts_unconfirmed_and_capacity(settings):
    g = snap(memory_used_mb=10000)
    g.unconfirmed_streams = 15  # 15 * 300 MB admitted but not yet visible in NVML
    assert not admit(g, StreamCost(memory_mb=300), settings).admitted
    g2 = snap()
    assert not admit(g2, StreamCost(memory_mb=300, max_streams=4), settings).admitted
    g2.avg_inference_latency_ms = 500
    assert "pipeline_latency" in admit(g2, StreamCost(memory_mb=300), settings).reason
    g3 = snap()
    g3.last_seen_age_s = 120
    assert "stale" in admit(g3, StreamCost(memory_mb=300), settings).reason


# ------------------------------------------------------------ events
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def emitter(published, eps=1000, clock=None):
    return EventEmitter(worker_id="W1", session_tag="abc", gie_models={1: {"model_id": "person", "model_version": "1"},
                                                                       3: {"model_id": "reid", "model_version": "1"}},
                        publish=lambda t, p, k: published.append((t, p, k)) or True, detection_interval=15,
                        embedding_interval=30, max_events_per_s=eps, clock=clock or Clock())


def frame(n, objs, w=1280, h=720):
    return FrameResult(source_id=0, frame_num=n, ntp_ns=1_700_000_000_000_000_000, src_width=w, src_height=h, objects=objs)


def obj(tid=7, emb=None):
    return ObjectResult(gie_id=1, class_id=0, label="person", confidence=0.91, tracker_id=tid, left=960, top=540,
                        width=192, height=108, embeddings={3: emb} if emb else {})


def test_emitter_interval_scaling_and_ids():
    pub = []
    e = emitter(pub)
    for n in range(0, 31):
        e.process("CAM_1", frame(n, [obj()]), 1920, 1080)
    dets = [p for t, p, _ in pub if t == "detection.events"]
    assert [d["frame_num"] for d in dets] == [0, 15, 30]
    assert dets[0]["event_type"] == "object_detected" and dets[1]["event_type"] == "object_tracked"
    assert dets[0]["object_id"] == "TRACK_abc_7"
    assert dets[0]["bbox"] == {"x": 640.0, "y": 360.0, "width": 128.0, "height": 72.0}  # scaled to 1280x720
    assert all(k == "CAM_1" for _, _, k in pub)


def test_emitter_untracked_and_embeddings():
    pub = []
    e = emitter(pub)
    e.process("CAM_1", frame(0, [obj(tid=UNTRACKED, emb=[3.0, 4.0])]), 1920, 1080)
    det = [p for t, p, _ in pub if t == "detection.events"][0]
    assert det["object_id"] is None
    assert not [p for t, p, _ in pub if t == "embedding.events"]  # untracked objects have no embedding identity
    pub.clear()
    e.process("CAM_1", frame(1, [obj(tid=9, emb=[3.0, 4.0])]), 1920, 1080)
    emb = [p for t, p, _ in pub if t == "embedding.events"][0]
    assert emb["embedding"] == pytest.approx([0.6, 0.8])


def test_emitter_rate_limit_drops_not_blocks():
    pub, clock = [], Clock()
    e = emitter(pub, eps=5, clock=clock)
    e.process("CAM_1", frame(0, [obj(tid=i) for i in range(20)]), 1920, 1080)
    assert len(pub) == 5 and e.dropped == 15
    clock.t = 1.0
    e.process("CAM_1", frame(1, [obj(tid=100 + i) for i in range(3)]), 1920, 1080)
    assert len(pub) == 8


def test_token_bucket_and_source_stats():
    c = Clock()
    b = TokenBucket(2, clock=c)
    assert b.take() and b.take() and not b.take()
    c.t = 0.5
    assert b.take()
    st = SourceStats(expected_fps=30)
    for i in range(91):
        st.on_frame(i / 15.0, 40.0)  # 15 fps actual for 6 s against 30 fps expected
    assert st.fps() == pytest.approx(15.0, rel=0.01)
    assert st.latency_ms() == 40.0
    assert st.deficit_frames == pytest.approx(75, abs=1)  # ~150 expected - 75 received in first 5 s window


def test_detection_interval_in_plan_signature_and_nvinfer_config():
    """Detector skip interval: primary nvinfer gets it, secondaries do not, and it changes the pipeline signature."""
    from app.services.model_router import PipelinePlan
    from worker.pipeline.config_gen import nvinfer_config

    def plan(interval):
        return PipelinePlan(stages=[], tracking_enabled=True, tracker_type="NvDCF", execution_strategy="independent",
                            detection_interval=interval)

    assert plan(0).signature != plan(2).signature          # cameras with different intervals never share a worker
    assert plan(2).to_dict()["detection_interval"] == 2
    st = dict(model_id="m", model_version="1", model_version_id=1, task="detector", backend="nvinfer", gie_unique_id=1,
              process_mode=1, precision="fp16", onnx_file="m.onnx", engine_file=None, labels=["a"], max_batch_size=2,
              nvinfer_properties={}, class_attrs={}, output_tensor_meta=False, operate_on_gie_id=None,
              operate_on_class_ids=None)
    assert "interval=3\n" in nvinfer_config(st, 0, "/models", None, interval=3)
    assert "interval=0\n" in nvinfer_config(st, 0, "/models", None)
    st2 = dict(st, process_mode=2, operate_on_gie_id=1)
    assert "interval=0\n" in nvinfer_config(st2, 0, "/models", None, interval=3)   # secondary unaffected
