"""Model Routing Engine.

Turns a camera's model assignments into an executable inference plan:

* Mode A - single model:           [PGIE(model)] -> tracker
* Mode B - multiple models:        [PGIE(m1), PGIE(m2), ...] -> tracker   (all on full frame)
* Mode C - cascade:                [PGIE(parent)] -> tracker -> [SGIE(child on parent's classes)]

Each plan has a deterministic ``signature``. Cameras whose plans share a signature
can be batched into the same DeepStream pipeline group (worker), which is how the
platform reuses loaded TensorRT engines instead of loading duplicates per camera.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.trackers import ALL_TRACKERS, is_python_tracker
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

TRACKERS = set(ALL_TRACKERS)


class RoutingError(ValueError):
    """Raised when a requested assignment cannot be executed."""


@dataclass
class InferenceStage:
    model_id: str
    model_version_id: int
    model_version: str
    task: str
    backend: str
    gie_unique_id: int
    process_mode: int  # 1 = primary (full frame), 2 = secondary (on objects)
    operate_on_gie_id: int | None = None
    operate_on_class_ids: list[int] | None = None
    parent_model_id: str | None = None
    labels: list[str] = field(default_factory=list)
    precision: str = "fp16"
    onnx_file: str | None = None
    engine_file: str | None = None
    int8_calib_file: str | None = None
    max_batch_size: int = 2
    nvinfer_properties: dict[str, Any] = field(default_factory=dict)
    class_attrs: dict[str, Any] = field(default_factory=dict)
    nvinferserver_config_file: str | None = None
    embedding_dim: int | None = None
    output_tensor_meta: bool = False
    detect_class_ids: list[int] | None = None  # classes this camera wants (None = all the model offers)
    conf_threshold: float | None = None        # per-camera minimum confidence (None = model default)
    iou_threshold: float | None = None         # per-camera NMS IoU (None = model default)


@dataclass
class PipelinePlan:
    stages: list[InferenceStage]
    tracking_enabled: bool
    tracker_type: str
    execution_strategy: str
    detection_interval: int = 0  # frames the detector skips between runs; tracking runs on every frame

    @property
    def primaries(self) -> list[InferenceStage]:
        return [s for s in self.stages if s.process_mode == 1]

    @property
    def secondaries(self) -> list[InferenceStage]:
        return [s for s in self.stages if s.process_mode == 2]

    @property
    def model_set(self) -> list[dict[str, Any]]:
        return [{"model_id": s.model_id, "version": s.model_version, "mode": s.process_mode,
                 "parent": s.parent_model_id} for s in self.stages]

    @property
    def signature(self) -> str:
        canon = {
            "stages": [
                [s.model_id, s.model_version_id, s.process_mode, s.parent_model_id,
                 sorted(s.operate_on_class_ids or [])]
                + (["detect", sorted(s.detect_class_ids)] if s.detect_class_ids else [])
                + (["conf", s.conf_threshold] if s.conf_threshold else [])
                + (["iou", s.iou_threshold] if s.iou_threshold else [])
                for s in self.stages
            ],
            "tracking": self.tracking_enabled,
            "tracker": self.tracker_type if self.tracking_enabled else None,
            "interval": self.detection_interval,
        }
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:32]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [asdict(s) for s in self.stages],
            "tracking_enabled": self.tracking_enabled,
            "tracker_type": self.tracker_type,
            "execution_strategy": self.execution_strategy,
            "detection_interval": self.detection_interval,
            "signature": self.signature,
        }


def _active_version(model: Model, versions: dict[int, ModelVersion]) -> ModelVersion:
    if model.active_version_id is None or model.active_version_id not in versions:
        raise RoutingError(f"model '{model.id}' has no active version")
    v = versions[model.active_version_id]
    if v.status != ModelVersionStatus.ACTIVE:
        raise RoutingError(f"model '{model.id}' active version {v.version} is {v.status.value}")
    if model.backend == InferenceBackend.NVINFER and not (v.onnx_file or v.engine_file):
        raise RoutingError(f"model '{model.id}' v{v.version} has neither onnx_file nor engine_file")
    if model.backend == InferenceBackend.NVINFERSERVER and not v.nvinferserver_config_file:
        raise RoutingError(f"model '{model.id}' v{v.version} requires nvinferserver_config_file")
    return v


def validate_assignment_request(
    requested: list[dict[str, Any]], models: dict[str, Model], strategy: ExecutionStrategy
) -> None:
    """Validate an API assignment request before it is persisted."""
    ids = [r["model_id"] for r in requested]
    if len(ids) != len(set(ids)):
        raise RoutingError("duplicate model_id in request")
    enabled = {r["model_id"] for r in requested if r.get("enabled", True)}
    for r in requested:
        m = models.get(r["model_id"])
        if m is None:
            raise RoutingError(f"unknown model '{r['model_id']}'")
        parent = r.get("parent_model_id")
        if parent:
            if strategy != ExecutionStrategy.CASCADE:
                raise RoutingError("parent_model_id requires execution_strategy='cascade'")
            if parent == r["model_id"]:
                raise RoutingError("a model cannot cascade on itself")
            if parent not in models:
                raise RoutingError(f"unknown parent model '{parent}'")
            if r.get("enabled", True) and parent not in enabled:
                raise RoutingError(f"parent model '{parent}' of '{r['model_id']}' must be assigned and enabled")
            if models[parent].task != ModelTask.DETECTOR:
                raise RoutingError(f"parent model '{parent}' must be a detector")
        elif m.task in (ModelTask.CLASSIFIER, ModelTask.REID) and r.get("enabled", True):
            raise RoutingError(f"model '{m.id}' ({m.task.value}) must cascade on a detector (set parent_model_id)")
    # cycle check: parents are single-level detectors so a chain deeper than one is
    # allowed only through detectors; detect cycles explicitly anyway
    graph = {r["model_id"]: r.get("parent_model_id") for r in requested}
    for start in graph:
        seen, cur = set(), start
        while cur is not None:
            if cur in seen:
                raise RoutingError("cyclic cascade definition")
            seen.add(cur)
            cur = graph.get(cur)


def build_plan(
    camera: Camera,
    assignments: list[CameraModelAssignment],
    models: dict[str, Model],
    versions: dict[int, ModelVersion],
    batch_size: int,
) -> PipelinePlan:
    """Build the inference plan for one camera. Raises RoutingError if not runnable."""
    active = [a for a in sorted(assignments, key=lambda a: (a.order, a.model_id)) if a.enabled]
    if not active:
        raise RoutingError("camera has no enabled model assignments")
    if camera.tracker_type not in TRACKERS:
        raise RoutingError(f"unsupported tracker '{camera.tracker_type}'")

    for a in active:
        m = models.get(a.model_id)
        if m is None:
            raise RoutingError(f"unknown model '{a.model_id}'")
        if not m.enabled:
            raise RoutingError(f"model '{m.id}' is disabled")

    cascade = camera.execution_strategy == ExecutionStrategy.CASCADE
    primaries = [a for a in active if not (cascade and a.parent_model_id)]
    secondaries = [a for a in active if cascade and a.parent_model_id]
    if not primaries:
        raise RoutingError("at least one primary (full-frame) model is required")

    stages: list[InferenceStage] = []
    gie_of: dict[str, int] = {}

    def make(a: CameraModelAssignment, mode: int, gie: int) -> InferenceStage:
        m = models[a.model_id]
        v = _active_version(m, versions)
        st = InferenceStage(
            model_id=m.id, model_version_id=v.id, model_version=v.version, task=m.task.value,
            backend=m.backend.value, gie_unique_id=gie, process_mode=mode, labels=list(v.labels or []),
            precision=v.precision, onnx_file=v.onnx_file, engine_file=v.engine_file,
            int8_calib_file=v.int8_calib_file,
            max_batch_size=batch_size if mode == 1 else max(v.max_batch_size, batch_size),
            nvinfer_properties=dict(v.nvinfer_properties or {}), class_attrs=dict(v.class_attrs or {}),
            nvinferserver_config_file=v.nvinferserver_config_file, embedding_dim=v.embedding_dim,
            output_tensor_meta=m.task in (ModelTask.REID, ModelTask.POSE)
            or bool((v.nvinfer_properties or {}).get("output-tensor-meta")),
            detect_class_ids=sorted(set(a.detect_class_ids)) if getattr(a, "detect_class_ids", None) else None,
            conf_threshold=getattr(a, "conf_threshold", None), iou_threshold=getattr(a, "iou_threshold", None),
        )
        if v.precision == "int8" and not v.int8_calib_file and not v.engine_file:
            raise RoutingError(f"model '{m.id}' int8 precision requires int8_calib_file or a prebuilt engine")
        return st

    gie = 1
    for a in primaries:
        st = make(a, 1, gie)
        gie_of[a.model_id] = gie
        stages.append(st)
        gie += 1
    # secondaries may depend on other secondaries only through detectors; resolve in order
    pending = list(secondaries)
    while pending:
        progressed = False
        for a in list(pending):
            if a.parent_model_id in gie_of:
                st = make(a, 2, gie)
                st.operate_on_gie_id = gie_of[a.parent_model_id]
                st.operate_on_class_ids = list(a.operate_on_class_ids) if a.operate_on_class_ids else None
                st.parent_model_id = a.parent_model_id
                gie_of[a.model_id] = gie
                stages.append(st)
                gie += 1
                pending.remove(a)
                progressed = True
        if not progressed:
            missing = ", ".join(f"{a.model_id}->{a.parent_model_id}" for a in pending)
            raise RoutingError(f"cascade parent not enabled/assigned: {missing}")

    return PipelinePlan(stages=stages, tracking_enabled=camera.tracking_enabled,
                        tracker_type=camera.tracker_type, execution_strategy=camera.execution_strategy.value,
                        # python trackers (ByteTrack, OC-SORT, ...) need a detection on every frame, so the interval is forced to 0
                        detection_interval=0 if is_python_tracker(camera.tracker_type)
                        else int(getattr(camera, "detection_interval", 0) or 0))
